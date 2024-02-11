import pandas as pd
import matplotlib.pyplot as plt

class Graphics:
    def plot_prediction_vs_identity_for_person(person_number, test_pred, test_rmse):
        # Create a DataFrame with the predicted and original values for ShankAngles and ThighAngles
        df = pd.DataFrame({'predicted_person': test_pred, 'identity': range(1, 101)})

        # Create a scatter plot with predicted and original values on the x-axis, and ShankAngles and ThighAngles on the y-axis
        plt.scatter(df['identity'], df['identity'], label='Identity Function'.format(person_number), alpha=0.3, marker='o', s=40, color='green', edgecolors='k')
        plt.scatter(df['identity'], df['predicted_person'], label='Predicted Gait % for Person {}'.format(person_number), alpha=0.7, marker='o', s=40, color='orange', edgecolors='k')

        plt.ylabel('Gait %')
        plt.xlabel('Predicted Gait %')
        plt.title('Prediction vs Identity for Person {}'.format(person_number))

        # Add RMSE to the plot
        plt.text(0.05, 0.95, f"RMSE: {round(test_rmse, 2)}", transform=plt.gca().transAxes)

        plt.legend()
        plt.savefig('person_{}.png'.format(person_number))
        plt.clf() # Clear the plot

