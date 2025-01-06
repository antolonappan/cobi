from solat_cb.mle import MLE
import numpy as np


class LowStat:

    def __init__(self, mle, n=100) -> None:
        self.mle = mle
        self.n = n

    def estimated_params(self):
        result = {k: [] for k in self.mle.estimate_angles(0).keys() if not k.startswith('A')}
        
        for i in range(self.n):
            res = self.mle.estimate_angles(i)
            for k, v in res.items():
                if not k.startswith('A'):
                    result[k].append(v)
        
        new_res = {k: {'mean': np.mean(v, axis=0), 'std': np.std(v, axis=0)} for k, v in result.items()}
        
        return new_res
    
    def fiducial_params(self):
        result = {}
        angles = self.mle.estimate_angles(0)
        
        for k in angles.keys():
            if not k.startswith('A'):
                if k == 'beta':
                    result[k] = self.mle.spec.lat.beta
                else:
                    try:
                        i = np.where(self.mle.spec.lat.freqs == k)[0][0]
                        result[k] = self.mle.spec.lat.alpha[i]
                    except IndexError:
                        print(f"Frequency {k} not found in spec.lat.freqs")
        return result
    
    def multipole_range(self,text=True):
        if text:
            return f"{self.mle.bmin}-{self.mle.bmax}"
        else:
            return self.mle.bmin, self.mle.bmax
    
    @property
    def bands(self):
        return self.mle.spec.lat.freqs

        


class MultiStat:

    def __init__(self, mledict, n=100) -> None:
        self.mledict = mledict
        self.n = n
        self.cases = list(mledict.keys())
        
    def extract_data(self):
        fiducial = None
        bands = None
        result = {}
        for case in self.cases:
            result[case] = {}
            mleobjs = self.mledict[case]
            for mleobj in mleobjs:
                ls = LowStat(mleobj, self.n)
                if fiducial is None:
                    fiducial = ls.fiducial_params()
                    bands = ls.bands
                else:
                    assert fiducial == ls.fiducial_params(), "Fiducial parameters do not match, This stat only works for same fiducial parameters"
                tag = ls.multipole_range(text=True)
                result[case][tag] = ls.estimated_params()
        result['fiducial'] = fiducial
        return result
    
    def plot(self):
        import matplotlib.pyplot as plt

        data = self.extract_data()
        fiducial = data['fiducial']
        del data['fiducial']

        _, axs = plt.subplots(len(fiducial), 1, figsize=(1 * len(data), 6), sharex=True)

        for idx, (key, fid_value) in enumerate(fiducial.items()):
            ax = axs[idx]
            ax.set_ylabel(key)
            ax.axhline(fid_value, color='r', linestyle='--')

            xtick_labels = []  # To store the xticks labels
            x_positions = []   # To store the x positions for the ticks

            for case_idx, case in enumerate(data.keys()):
                case_data = data[case]
                subkeys = case_data.keys()

                # Set up x positions, adjusting for case_idx with spacing
                case_x_positions = apply_diff(np.ones(len(subkeys)) * case_idx, 0.1)
                x_positions.extend(case_x_positions)

                # Create xticks depending on the length of subkeys
                if len(subkeys) == 1:
                    xtick_labels.append(case)  # Just use data.keys() if there's only one subkey
                else:
                    for subkey in subkeys:
                        xtick_labels.append(f"{case}_{subkey}")  # Include subkey if more than one

                mean, std = [], []
                for subkey in subkeys:
                    mean.append(case_data[subkey][key]['mean'])
                    std.append(case_data[subkey][key]['std'])
                ax.errorbar(case_x_positions, mean, yerr=std, fmt='o')

            # Set xticks and xticklabels only once at the last iteration
            if idx == len(fiducial) - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

        

def apply_diff(a,diff):
    n = len(a)
    match n:
        case 1:
            return a
        case 2:
            return [a[0] - diff, a[1] + diff]
        case _:
            central_idx = n // 2  # Index of the central value
            central_value = a[central_idx]
            for i in range(1, central_idx + 1):
                a[central_idx - i] -= diff * i
                a[central_idx + i] += diff * i
            return a
    


        

        