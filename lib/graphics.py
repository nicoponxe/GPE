import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DPI = 800
        
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

    def plot_kalman_vs_original(df, person_number, include_shank_angles, include_thigh_angles):
        if include_shank_angles:
            plt.scatter(df['gait_percentage'], df['ShankAnglesOriginal'], label='Original Shank Angles', alpha=0.5, marker='o', s=5, color='blue', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ShankAngles'], label='Kalman Filtered Shank Angles', alpha=1.0, marker='o', s=5, color='orange', edgecolors='k')

        if include_thigh_angles:
            plt.scatter(df['gait_percentage'], df['ThighAnglesOriginal'], label='Original Thigh Angles', alpha=0.5, marker='o', s=5, color='green', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ThighAngles'], label='Kalman Filtered Thigh Angles', alpha=1.0, marker='o', s=5, color='red', edgecolors='k')

        plt.ylabel('Angle (degrees)')
        plt.xlabel('Gait %')
        plt.title('Kalman Filter vs Original for Person {}'.format(person_number))

        plt.legend()
        plt.savefig('kalman_vs_original_angles_{}.png'.format(person_number), dpi=OUTPUT_DPI)
        plt.clf()

        if include_shank_angles:
            plt.scatter(df['gait_percentage'], df['ShankAngularVelocityOriginal'], label='Original Shank Angles', alpha=0.7, marker='o', s=40, color='blue', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ShankAngularVelocity'], label='Kalman Filtered Shank Angles', alpha=0.7, marker='o', s=40, color='orange', edgecolors='k')

        if include_thigh_angles:
            plt.scatter(df['gait_percentage'], df['ThighAngularVelocityOriginal'], label='Original Thigh Angles', alpha=0.7, marker='o', s=40, color='green', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ThighAngularVelocity'], label='Kalman Filtered Thigh Angles', alpha=0.7, marker='o', s=40, color='red', edgecolors='k')

        plt.ylabel('Angle (degrees)')
        plt.xlabel('Gait %')
        plt.title('Kalman Filter vs Original for Person {}'.format(person_number))

        plt.legend()
        plt.savefig('kalman_vs_original_angular_velocity_{}.png'.format(person_number), dpi=OUTPUT_DPI)
        plt.clf()
