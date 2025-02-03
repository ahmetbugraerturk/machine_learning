from linear_regression import model
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/chicago_taxi_train.csv")

train_df = data[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

# target encoding
train_df['COMPANY'] = train_df.groupby('COMPANY')['FARE'].transform('mean')
train_df['PAYMENT_TYPE'] = train_df.groupby('PAYMENT_TYPE')['FARE'].transform('mean')


print(train_df.describe(include="all"))
print(train_df.nunique())

print(train_df.corr(numeric_only=True)) # Most correlated: TRIP_MILES, least correlated: TIP_RATE

sns.pairplot(train_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS", 'TIP_RATE', 'COMPANY', 'PAYMENT_TYPE'], y_vars=["FARE"])
plt.show()


M1 = model(train_df, ['TRIP_MILES'], "FARE", 0.001, 50, 50, types_of_loss="mse", plotly_renderer="browser")
M1.plot_2d_combined()

M2 = model(train_df, ['TRIP_MILES', 'TRIP_SECONDS'], "FARE", 0.001, 50, 50, types_of_loss="mse", plotly_renderer="browser")
M2.plot_3d_combined()

# Note: TIP_RATE is not strongly correlated, but this is for experimentation
M3 = model(train_df, ['TRIP_MILES', 'TRIP_SECONDS', 'TIP_RATE'], "FARE", 0.001, 50, 50, types_of_loss="mse", plotly_renderer="browser")
M3.plot_loss_curve()

# For categorical var.
M4 = model(train_df, ['TRIP_MILES', "PAYMENT_TYPE"], "FARE", 0.001, 50, 50, types_of_loss="mse", plotly_renderer="browser")
M4.plot_3d_combined()

print(M1)
print(M2)
print(M3)
print(M4)



