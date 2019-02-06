import numpy as np
import matplotlib.pyplot as plt

import argparse

def main(args):
    colors = 'rgbymp'
    for i,res in enumerate(args.results):
        with open(res,'rb') as f:
            data = np.load(f, encoding='latin1')
            dt = data['dt']
            length = data['length']
            times = dt*np.arange(length)
            plt.plot(times, data['mean_dist'], colors[i], label=data['model'])
            plt.fill_between(times, data['mean_dist'] - data['std_dist'], 
                    data['mean_dist'] + data['std_dist'], color=colors[i], alpha=0.3)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', help="File with the stored results")
    args = parser.parse_args()
    main(args)
