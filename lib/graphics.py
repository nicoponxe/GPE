import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DPI = 600
        
class Graphics:
    def plot_prediction_vs_identity_for_person(person_number, test_pred, test_rmse, total_number_of_people = 21):
        subject_number = (person_number % total_number_of_people) + 1
        speeds = ['0.5', '1.0', '1.5']
        speed_index = person_number // (total_number_of_people + 1)

        # Create a DataFrame with the predicted and original values for ShankAngles and ThighAngles
        df = pd.DataFrame({'predicted_person': test_pred, 'identity': range(1, 101)})

        # Create a scatter plot with predicted and original values on the x-axis, and ShankAngles and ThighAngles on the y-axis
        plt.scatter(df['identity'], df['identity'], label='Identity Function'.format(person_number), alpha=0.3, marker='o', s=40, color='green', edgecolors='k')
        plt.scatter(df['identity'], df['predicted_person'], label='Predicted Gait % for SUbject {} @ {}'.format(subject_number, speeds[speed_index]), alpha=0.7, marker='o', s=40, color='orange', edgecolors='k')

        plt.ylabel('Gait %')
        plt.xlabel('Predicted Gait %')
        plt.title('Prediction vs Identity for Subject {} @ {}'.format(subject_number, speeds[speed_index]))

        # Add RMSE to the plot
        plt.text(0.05, 0.95, f"RMSE: {round(test_rmse, 2)}", transform=plt.gca().transAxes)

        plt.legend()
        plt.savefig('person_{}_{}.png'.format(subject_number, speeds[speed_index]), dpi=OUTPUT_DPI)
        plt.clf() # Clear the plot

    def plot_kalman_vs_original(df, person_number, include_shank_angles, include_thigh_angles, total_number_of_people = 21):
        subject_number = (person_number % total_number_of_people)
        speeds = ['0.5', '1.0', '1.5']
        speed_index = (subject_number - 1) % 3  # -1 because person_number goes from 1 to 21

        if include_shank_angles:
            plt.scatter(df['gait_percentage'], df['ShankAnglesOriginal'], label='Original Shank Angles', alpha=0.5, marker='o', s=5, color='blue', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ShankAngles'], label='Kalman Filtered Shank Angles', alpha=1.0, marker='o', s=5, color='orange', edgecolors='k')

        if include_thigh_angles:
            plt.scatter(df['gait_percentage'], df['ThighAnglesOriginal'], label='Original Thigh Angles', alpha=0.5, marker='o', s=5, color='green', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ThighAngles'], label='Kalman Filtered Thigh Angles', alpha=1.0, marker='o', s=5, color='red', edgecolors='k')

        plt.ylabel('Angle (degrees)')
        plt.xlabel('Gait %')
        plt.title('Kalman Filter vs Original for Subject {} @ {} m/s'.format(subject_number, speeds[speed_index]))

        plt.legend()
        plt.savefig('kalman_vs_original_angles_{}_{}.png'.format(subject_number, speeds[speed_index]), dpi=OUTPUT_DPI)
        plt.clf()

        if include_shank_angles:
            plt.scatter(df['gait_percentage'], df['ShankAngularVelocityOriginal'], label='Original Shank Angles', alpha=0.7, marker='o', s=40, color='blue', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ShankAngularVelocity'], label='Kalman Filtered Shank Angles', alpha=0.7, marker='o', s=40, color='orange', edgecolors='k')

        if include_thigh_angles:
            plt.scatter(df['gait_percentage'], df['ThighAngularVelocityOriginal'], label='Original Thigh Angles', alpha=0.7, marker='o', s=40, color='green', edgecolors='k')
            plt.scatter(df['gait_percentage'], df['ThighAngularVelocity'], label='Kalman Filtered Thigh Angles', alpha=0.7, marker='o', s=40, color='red', edgecolors='k')

        plt.ylabel('Angular Velocity')
        plt.xlabel('Gait %')
        plt.title('Kalman Filter vs Original for Subject {} @ {} m/s'.format(subject_number, speeds[speed_index]))

        plt.legend()
        plt.savefig('kalman_vs_original_angular_velocity_{}_{}.png'.format(subject_number, speeds[speed_index]), dpi=OUTPUT_DPI)
        plt.clf()
