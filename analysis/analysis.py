import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def plot_player_trajectory(self):
        """
        This example assumes processed data with keys: 'x', 'y', 'episode'
        """
        plot_dict = {}
        import os
        import json
        
        for file in os.listdir(self.save_dir):
            if file.endswith(".json"):
                filepath = os.path.join(self.save_dir, file)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                for episode in data['trajectories']:
                    xs = [pos['x'] for pos in episode]
                    ys = [pos['y'] for pos in episode]
                    ax.plot(xs, ys, marker='o', alpha=0.7)
                ax.set_title("Player Trajectory in Map")
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
                plot_dict[file+'_trajectory'] = fig
        
        return plot_dict