import numpy as np

from typing import Callable

from tensorflow import keras


class SlabSystem:
    def __init__(self, system_type="DEV-IM"):
        self.select(system_type)

    # Select from a set of predefined systems
    def select(self, system_type):
        match system_type:
            case "IMAS":
                self.set_specs(8, 8, 1, 5, 5 * 24 * 16, 26.0)
            case "IMAS-1ring":
                self.set_specs(8, 8, 1, 1,     24 * 16, 26.0)
            case "EBRAIN":
                self.set_specs(16, 8, 2, 8*40, 26.0)
            case "ERC":
                self.set_specs(24, 8, 3, 4*20, 26.0)
            case "DEV-IM":
                self.set_specs(8, 8, 1, 0, 16, 26.0) # Dev IMAS system, with just one supermodule
            case "DEV-EB":
                self.set_specs(16, 8, 2, 0, 16, 26.0) # Dev EBRAINS system, with just one supermodule
            case _:
                self.set_specs(8, 8, 1, 5, 24*5*16, 26.0) # Defaults to IMAS

    # Manually set system specification
    def set_specs(self, slabs, SiPM, slabs_per_si, rings, mmodules, width, groups=0):
        self.N_SLABS = slabs # Number of cols (slabs).
        self.N_SiPM = SiPM   # Number of rows (silicons per slab).
                             # These are the energy channels
        self.SLABS_PER_SIPM = slabs_per_si # Number of slabs per silicon (silicon cols).
                                           # Time channels are N_SLABS/SLABS_PER_SIPM
        self.N_RINGS = rings # Number of rings
        self.N_MODULES = mmodules # Total number of minimodules (for all rings)
        self.crystal_width = width

        self.slab_size = width/slabs
        self.sipm_size = width/SiPM

        self.slab_pos = self.slab_size/2 + np.arange(0, slabs, 1)*self.slab_size
        self.sipm_pos = self.sipm_size/2 + np.arange(0, SiPM, 1)*self.sipm_size

        self.set_grouping(groups) # Sets default grouping

    # Defines group mapping (categories used on training and inference) based on slab grouping
    def set_grouping(self, groups):
        total_slabs = self.N_SLABS*self.N_MODULES # Total number of slabs in system (for a supermodule, it is N_SLABS*16)
        sortedlist = np.arange(total_slabs) # Create the array with the number of slabs to be filled with mapping indexes

        match groups:
            case 0: # No grouping, each slab has its own category, which is unique for all slabs in the system
                    # Slab category is actually slab index
                mapping = sortedlist
            case 1: # SiPM grouping. All slabs sharing SiPM are grouped in the same category.
                    # Total number of categories is divided by the number of slabs in a SiPm
                mapping = np.floor_divide(sortedlist, self.SLABS_PER_SIPM)
            case 2: # MM grouping. Each minimodule has its own category, all the slabs in the
                    # same MM are grouped together.
                mapping = np.floor_divide(sortedlist, self.N_SLABS)
            case 3: # Position grouping. Each slab on the same position is in the same category.
                    # This means that there are a total of N_SLABS groups
                mapping = np.remainder(sortedlist, self.N_SLABS)
            case _: # Default to no grouping
                mapping = sortedlist

        self.set_mapping(mapping)

    # Manually set individual slab mapping using an array
    def set_mapping(self, mapping):
        self.group_mapping = mapping


# Class definition for a neural network used on slab-based system
# training and inference. Training is done using an embedding model,
# Inference will use the previously trained model or a manually loaded
# model. Make sure dataset format is adequate when training or inferring
# data
class SlabNN:
    # Create class associated with a slab system. NN model not yet defined
    def __init__(self, system: SlabSystem):
        self.slab_system = system
        self.model       = None

    # Create a combined model, using pre-trained models for Y and DOI. This model is not
    # intended to be trained
    def build_combined(self, XY_model_file, DOI_model_file, print_model=True, name='Combined'):
        model_XY  = keras.models.load_model(XY_model_file)

        model_DOI = keras.models.load_model(DOI_model_file)

        concatenated_output = keras.layers.Concatenate()([model_XY.layers[-1].output, model_DOI.layers[-1].output])

        model_name = name

        self.model = keras.Model(inputs  = [(model_XY.inputs, model_DOI.inputs)],
                                 outputs = concatenated_output,
                                 name    = model_name)

        if (print_model == True):
            self.model.summary()
            keras.utils.plot_model(self.model, to_file=self.model.name + "-model.png", show_shapes=True)

    # Train the model with the provided dataset
    def train(self, input_dataset, input_labels, val_dataset, val_labels, epochs=25, batch=1024):
        # Make sure model exists
        if (self.model is None):
            print ("SlabNN ERROR: Model not yet defined, please define model prior to training")
            return

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, min_delta=0.005)

        print ("SlabNN: Training NN model...")
        # Train model using provided datasets
        history = self.model.fit(
            x=input_dataset,
            y=input_labels,
            epochs=epochs,
            validation_data=(val_dataset, val_labels),
            batch_size=batch,
            shuffle=True,
            verbose=1,
            callbacks=[early_stop])

        return history

    # Predict the output values with the existing model, using the provided dataset
    def predict(self, dataset, batch=16384, verbosity=0):
        # Make sure model exists
        if (self.model is None):
            print ("SlabNN ERROR: Model not yet defined, please define model prior to predicting")
            return

        # Predict output values
        # predictions = self.model.predict(dataset, batch_size=batch, verbose=verbosity)
        predictions = self.model(dataset, training=False)

        # Option B
        if (self.model.name == "Combined"):
            predicted_values_Y   = predictions[:,0]
            predicted_values_DOI = predictions[:,1]
        else:
            predicted_values_Y   = predictions
            predicted_values_DOI = None

        return predicted_values_Y, predicted_values_DOI

    # Save existing model to a file
    def save(self, file):
        self.model.save(file)

    # Load existing model from a file
    def load(self, file):
        self.model = keras.models.load_model(file)


def neural_net_pcalc(system   : str     ,
                     y_file   : str     ,
                     doi_file : str     ,
                    #  mm_indx  : Callable,
                     local_pos: Callable
                     ) -> Callable:
    """
    Defines the desired neural network and
    uses the model to predict SM level
    impact positions.
    """
    NN = SlabNN(SlabSystem(system))
    NN.build_combined(y_file, doi_file, print_model=False)
    ## Dummies for event by event for now.
    # categories = np.zeros(1, np.int32)
    #slab_max = select_max_energy(ChannelType.TIME)
    def _xy_local(slab_id: int, y_mm: np.float32) -> tuple[float, float]:
        local_x, slab_y = local_pos(slab_id)
        local_y         = y_mm + slab_y
        return local_x, local_y

    def _predict(slab_ids: np.ndarray,
                 evts    : np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        # Cats all 0 for now.
        categories = np.zeros_like(evts['slab_idx'])
        #energies = np.zeros((1, 8), np.float32)
        # for imp in filter(lambda x: x[1] is ChannelType.ENERGY, sm_info):
        #     energies[0][mm_indx(imp[0])] = imp[3]
        mm_y, doi = NN.predict([categories, evts['Esignals'], categories, evts['Esignals']])
        # print(f'DOI is {doi}, types {type(mm_y)}')
        local_pos = np.asarray(tuple(map(_xy_local, slab_ids, mm_y)))
        return local_pos, doi
        # max_slab = slab_max(sm_info)[0]
        # local_x, slab_y = local_pos(max_slab)
        # local_y         = mm_y[0] + slab_y
        # return local_x, local_y, doi[0]
    return _predict
