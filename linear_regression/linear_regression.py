import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class model:
    """
    A linear regression model with options for normalization and loss functions.
    
    Attributes
    ----------
    df : DataFrame
        The dataset used for training the model.
    features : list
        List of feature column names in the dataset.
    label : str
        The target label (dependent variable).
    learning_rate : float
        The learning rate for gradient descent optimization.
    epochs : int
        The number of training epochs.
    batch_size : int
        The size of each mini-batch for stochastic gradient descent.
    types_of_loss : str
        The type of loss function ('mse' or 'mae').
    normalization : str
        The normalization method ('minmax' or 'standard').
    plotly_renderer : str 
        Plotly renderer ('plotly_mimetype', 'jupyterlab', 'nteract', 'vscode', 'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab', 'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe', 'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png')
    
    Methods
    -------
    prediction(x)
        Makes predictions for input features x using the learned weights and bias.
    plot_loss_curve()
        Plots the loss curve (MSE or MAE) over the epochs of training.
    plot_2d_predictions()
        Plots a 2D graph of actual vs. predicted values when the model has only one feature.
    plot_2d_combined()
        Plots both the loss curve and the actual vs. predicted values in a single 2D combined plot when the model has only one feature.
    plot_3d_predictions()
        Creates a 3D surface plot of the predicted vs. actual values when the model has two features.
    plot_3d_combined()
        Plots the loss curve and 3D prediction surface in separate plots.
    """
    def __init__(self, df, features, label, learning_rate, epochs, batch_size, types_of_loss = "mse", normalization = "standard", plotly_renderer = "notebook_connected"):
        """
        Initializes the model with dataset, features, target label, hyperparameters, and normalization method.
        
        Parameters:
        df (DataFrame): The dataset used for training.
        features (list): List of feature column names.
        label (str): The target label column name.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
        batch_size (int): Size of mini-batches.
        types_of_loss (str): Loss function type ('mse' or 'mae').
        normalization (str): Normalization method ('minmax' or 'standard').
        plotly_renderer (str): Plotly renderer ('plotly_mimetype', 'jupyterlab', 'nteract', 'vscode', 'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab', 'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe', 'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png')
        """
        
        # adjust plotly graphs renderer
        pio.renderers.default = plotly_renderer
        
        #initialize
        self.df = df.copy()
        self.features = features
        self.label = label
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.types_of_loss = types_of_loss
        self.normalization = normalization
        
        #to solve one of the error about dtype of a pandas object
        for feature in features + [label]:
            if pd.api.types.is_numeric_dtype(self.df[feature]):
                self.df[feature] = self.df[feature].astype('float64')
        
        #for normalizaiton
        self.__norm_mins = {}
        self.__norm_maxs = {}
        self.__norm_means = {}
        self.__norm_stds = {}
        
        #normalize datas
        self.__normalize_data()
        
        #init some variables
        self.__weights = np.random.randn(len(features))
        self.__bias = np.random.randn()
        self.__batches = [] # Mini-batch stochastic gradient descent (mini-batch SGD)
        self.__loss = []
        
        #run required functions
        self.__seperate_batches()
        self.__train()
        
        self.__denormalize_data()
        
        self.verify()
        
        
    
    def __seperate_batches(self):
        """
        Shuffles the dataset and splits it into mini-batches for training using mini-batch stochastic gradient descent.
        """
        shuffled_df = self.df.sample(frac=1)
        for i in range(len(shuffled_df)//self.batch_size+1):
            if i == len(shuffled_df)//self.batch_size:
                self.__batches.append(shuffled_df.iloc[range(self.batch_size*i, len(shuffled_df)), :].copy())
            else:
                self.__batches.append(shuffled_df.iloc[range(self.batch_size*i, self.batch_size*(i+1)), :].copy())
        
        print(f"dataFrame with {len(shuffled_df)} length was shuffled and seperated {len(self.__batches)} batches with batch size {self.batch_size}.")
        
    def __mae(self, predicted, actual):
        """
        Calculates the Mean Absolute Error (MAE) between predicted and actual values.
        
        Parameters:
        predicted (numpy.ndarray): The predicted values.
        actual (numpy.ndarray): The actual values.
        
        Returns:
        float: The calculated MAE.
        """
        return np.mean(np.abs(predicted-actual))
    
    def __mse(self, predicted, actual):
        """
        Calculates the Mean Squared Error (MSE) between predicted and actual values.
        
        Parameters:
        predicted (numpy.ndarray): The predicted values.
        actual (numpy.ndarray): The actual values.
        
        Returns:
        float: The calculated MSE.
        """
        return np.mean(np.square(predicted-actual))
    
    def prediction(self, x):
        """
        Makes predictions for input features x using the learned weights and bias.
        
        Parameters:
        x (numpy.ndarray): Input features.
        
        Returns:
        numpy.ndarray: Predicted values.
        """
        if len(self.features)==1:
            x = x.reshape(-1, 1) #to solve an error when running dot product with only 1 feature
        return np.dot(x, self.__weights) + self.__bias
        
    def __train(self):
        """
        Trains the model using mini-batch stochastic gradient descent.
        """
        for i in range(self.epochs):
            print(f"\n epoch {i+1}/{self.epochs}")
            epoch_loss = 0
            for i, batch in enumerate(self.__batches):
                
                feature_matrix = batch[self.features].values #to make features values a numpy arrays matrix
                label_values = batch[self.label].values #to make label values a numpy array
                predicted = self.prediction(feature_matrix) #to get predictions for current batch with current weights and bias
                
                #calculate loss and derive new weights and bias with using gradient descent
                if self.types_of_loss == "mse":
                    loss = self.__mse(predicted, label_values)
                    w_slope = np.mean((predicted - label_values).reshape(-1, 1) * 2 * feature_matrix, axis = 0)
                    b_slope = np.mean((predicted - label_values) * 2)
                    
                elif self.types_of_loss == "mae":
                    loss = self.__mae(predicted, label_values)
                    w_slope = np.mean(np.sign(predicted - label_values).reshape(-1, 1) * feature_matrix, axis = 0)
                    b_slope = np.mean(np.sign(predicted - label_values))
                
                self.__weights -= w_slope*self.learning_rate
                self.__bias -= b_slope*self.learning_rate
                
                #saving loss to show in a plot at the end of the training
                epoch_loss += loss
                
                # PROGRESS BAR (written by chatgpt)
                progress = (i + 1) / len(self.__batches)
                num_hashes = int(progress * 30)
                bar = "#" * num_hashes + "-" * (30 - num_hashes)
                print(f"\r[{bar}] {progress * 100:.1f}% | Loss: {loss:.4f} | Weight: {self.__weights} | Bias: {self.__bias:.4f}",end="", flush=True)                

            self.__loss.append(epoch_loss/len(self.__batches))
        
    def __normalize_data(self):
        """
        Normalizes the data using the specified normalization method ('minmax' or 'standard').
        """
        if self.normalization == "minmax":
            self.__minmax_normalize()
        elif self.normalization == "standard":
            self.__standard_normalize()
        else:
            print("No normalization applied")
            
    def __minmax_normalize(self):
        """
        Applies Min-Max normalization to the features in the dataset, scaling them to a [0, 1] range.
        """
        self.__norm_mins[self.label] = self.df[self.label].min()
        self.__norm_maxs[self.label] = self.df[self.label].max()
        self.df[self.label] = (self.df[self.label] - self.__norm_mins[self.label]) / (self.__norm_maxs[self.label] - self.__norm_mins[self.label])
        for feature in self.features:
            self.__norm_mins[feature] = self.df[feature].min()
            self.__norm_maxs[feature] = self.df[feature].max()
            
            # x_norm = (x - min) / (max - min)
            self.df[feature] = (self.df[feature] - self.__norm_mins[feature]) / (self.__norm_maxs[feature] - self.__norm_mins[feature])
    
    def __standard_normalize(self):
        """
        Applies standard normalization to the features in the dataset, scaling them to have zero mean and unit standard deviation.
        """
        self.__norm_means[self.label] = self.df[self.label].mean()
        self.__norm_stds[self.label] = self.df[self.label].std()
        self.df[self.label] = (self.df[self.label] - self.__norm_means[self.label]) / self.__norm_stds[self.label]
        for feature in self.features:
            self.__norm_means[feature] = self.df[feature].mean()
            self.__norm_stds[feature] = self.df[feature].std()
            
            # x_norm = (x - mean) / std
            self.df.loc[:,feature] = (self.df[feature] - self.__norm_means[feature]) / self.__norm_stds[feature]
        
    def __denormalize_data(self):
        """
        Denormalizes the data using the specified normalization method ('minmax' or 'standard').
        """
        if self.normalization == "minmax":
            self.__minmax_denormalize()
        elif self.normalization == "standard":
            self.__standard_denormalize()
        else:
            print("No denormalization applied")
            
    def __minmax_denormalize(self):
        """
        Applies denormalization for Min-Max normalization to the features in the dataset.
        """
        self.df[self.label] = (self.df[self.label] * (self.__norm_maxs[self.label] - self.__norm_mins[self.label]) + self.__norm_mins[self.label])
        self.__bias = self.__bias * (self.__norm_maxs[self.label] - self.__norm_mins[self.label]) + self.__norm_mins[self.label]
        for i, feature in enumerate(self.features):
            # x = (x_norm * (max - min) - min)
            self.df[feature] = (self.df[feature] * (self.__norm_maxs[feature] - self.__norm_mins[feature]) + self.__norm_mins[feature])
            self.__bias -= self.__weights[i] * self.__norm_mins[feature] * (self.__norm_maxs[self.label] - self.__norm_mins[self.label]) / (self.__norm_maxs[feature] - self.__norm_mins[feature])
            self.__weights[i] = self.__weights[i] * (self.__norm_maxs[self.label] - self.__norm_mins[self.label]) / (self.__norm_maxs[feature] - self.__norm_mins[feature])
        # For Min-Max normalization, denormalize using the label's min and max
        self.__loss = [loss * (self.__norm_maxs[self.label] - self.__norm_mins[self.label]) + self.__norm_mins[self.label] for loss in self.__loss]
    
    def __standard_denormalize(self):
        """
        Applies denormalization for standard normalization to the features in the dataset.
        """
        self.df[self.label] = self.df[self.label] * self.__norm_stds[self.label] + self.__norm_means[self.label]
        self.__bias = self.__bias * self.__norm_stds[self.label] + self.__norm_means[self.label]
        for i, feature in enumerate(self.features):
            # x = x_norm * std + mean
            self.df.loc[:,feature] = self.df[feature] * self.__norm_stds[feature] + self.__norm_means[feature]
            # Convert bias to original scale: bias_original = (bias_normalized * std_label) + mean_label - sum(weight_normalized * mean_feature * std_label / std_feature)
            self.__bias -= self.__weights[i] * self.__norm_means[feature] * self.__norm_stds[self.label] / self.__norm_stds[feature]
            # Convert weight to original scale: weight_original = (weight_normalized * std_label) / std_feature
            self.__weights[i] = self.__weights[i] * self.__norm_stds[self.label] / self.__norm_stds[feature]
        # For Standard normalization, denormalize using the label's mean and std
        self.__loss = [loss * self.__norm_stds[self.label] + self.__norm_means[self.label] for loss in self.__loss]

            
    #all the plot functions are written by chatgpt
    def plot_loss_curve(self):
        """
        Plots the loss curve (MSE or MAE) over the epochs of training.
        """
        label = self.types_of_loss
        if self.types_of_loss=="mse":
            loss = np.power(self.__loss, 1/2)
            label = "rmse"
        else:
            loss = self.__loss
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), loss, marker='.', linestyle='-', color='r', label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve ({label})")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_2d_predictions(self):
        """
        Plots a 2D graph of actual vs. predicted values when the model has only one feature.
        """
        if len(self.features) != 1:
            print("2D plot requires exactly 1 feature.")
            return
        
        sample_data = self.df.sample(200)
        feature = sample_data[self.features[0]].values
        actual = sample_data[self.label]
        predicted = self.prediction(feature)
        
        # Plot with original-scale values
        plt.scatter(feature, actual, label="Actual")
        plt.plot(feature, predicted, color="red", label="Predicted")
        plt.xlabel(self.features[0])
        plt.ylabel(self.label)
        plt.title("Actual vs Predicted (Original Scale)")
        plt.legend()
        plt.show()

    def plot_2d_combined(self):
        """
        Plots both the loss curve and the actual vs. predicted values in a single 2D combined plot when the model has only one feature.
        """
        if len(self.features)!=1:
            print("This model cannot be plotted in 2 dimensions, since it has not 2 axes.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 satır, 2 sütun (yan yana), 12x5 boyut
        
        label = self.types_of_loss
        if self.types_of_loss=="mse":
            loss = np.power(self.__loss, 1/2)
            label = "rmse"
        else:
            loss = self.__loss
        
        axes[0].plot(range(1, self.epochs + 1), loss, marker='.', linestyle='-', color='r', label="Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Loss Curve ({label})")
        axes[0].legend()
        axes[0].grid(True)
    
        sample_data = self.df.sample(200)
        feature = sample_data[self.features[0]].values
        actual = sample_data[self.label]
        predicted = self.prediction(feature)

        
        axes[1].scatter(sample_data[self.features[0]], actual, label="Actual Values", color="blue")
        axes[1].plot(sample_data[self.features[0]], predicted, label="Predicted Values", color="red")
        axes[1].set_xlabel(self.features[0])
        axes[1].set_ylabel(self.label)
        axes[1].set_title(f"Actual Values vs. Predicted Values ({self.label})")
        axes[1].legend()
        axes[1].grid(True)
    
        plt.tight_layout() 
        plt.show()
        
    def plot_3d_predictions(self):
        """
        Creates a 3D surface plot of the predicted vs. actual values when the model has two features.
        """
        if len(self.features)!=2:
            print("This model cannot be plotted in 3 dimensions, since it has not 3 axes.")
            return
        
        # Select a sample of the data
        sample_data = self.df.sample(200)
        actual_values = sample_data[self.label]
    
        # Create meshgrid for the feature space
        x_range = np.linspace(sample_data[self.features[0]].min(), sample_data[self.features[0]].max(), 50)
        y_range = np.linspace(sample_data[self.features[1]].min(), sample_data[self.features[1]].max(), 50)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        
        # Prepare feature grid for prediction
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        z_grid = self.prediction(grid_points)
        
        # Reshape z_grid for surface plotting
        z_grid = z_grid.reshape(x_grid.shape)
    
        # Create 3D plot
        fig = go.Figure()
    
        # Surface plot for predicted values
        fig.add_trace(go.Surface(
            x=x_range, 
            y=y_range, 
            z=z_grid,
            colorscale='Viridis',
            opacity=0.7,
            name='Predicted Surface'
        ))
    
        # Scatter plot for actual values
        fig.add_trace(go.Scatter3d(
            x=sample_data[self.features[0]], 
            y=sample_data[self.features[1]], 
            z=actual_values,
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.6),
            name='Actual'
        ))
    
        # Adding titles and labels
        fig.update_layout(
            title=f"3D Surface Plot of Actual vs Predicted ({self.label})",
            scene=dict(
                xaxis_title=self.features[0],
                yaxis_title=self.features[1],
                zaxis_title=self.label
            ),
            showlegend=True
        )
        
        fig.show()
        
    def plot_3d_combined(self):
        """
        Plots the loss curve and 3D prediction surface in separate plots.
        """
        if len(self.features)!=2:
            print("This model cannot be plotted in 3 dimensions, since it has not 3 axes.")
            return
        
        print("Since 3D plot is made by plotly, they cannot be combined; therefore, this function plot these graphs separately.")
        self.plot_loss_curve()
        self.plot_3d_predictions()
        
    def verify(self):
        print("\n###### -*- VERIFICATION -*- ######\n")
        print("Final loss (original scale):\n")
        if self.types_of_loss=="mse":
            loss = self.__loss[-1]**(1/2)
        else:
            loss = self.__loss[-1]
        print(loss)
        #to print the equation at the end of the training
        print("\nFinal equation (original scale):\n")
        equation = f"{self.label} = "
        for i, feature in enumerate(self.features):
            equation += f"{self.__weights[i]:.4f} * {feature} + "
        self.__bias
        equation += f"{self.__bias:.4f}"
        print(equation)
        
        sample_data = self.df.sample(200)
        print("\nFeatures:")
        print(sample_data[self.features].head())
        
        print("\nLabel:")
        print(sample_data[self.label].head())
        
        normalized_predictions = self.prediction(sample_data[self.features].values)
        
        
        print("\nPredictions vs Actual values:")
        for i in range(5):  # Print first 5 samples
            print(f"Predicted: {normalized_predictions[i]:.2f}, Actual: {sample_data[self.label].values[i]:.2f}")
        print()
        
    def __str__(self):
        equation = f"{self.label} = "
        for i, feature in enumerate(self.features):
            equation += f"{self.__weights[i]:.4f} * {feature} + "
        self.__bias
        equation += f"{self.__bias:.4f}"
        return equation

        