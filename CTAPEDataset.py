from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from math import isnan

SAMPLE_ID = ["Sample ID", "Sample #"]
WAVELENGTH = ["Wavelength (nm)", "Wavelength"]


class CTAPEDataset(Dataset):
    wavelength = []
    sample_id = "Sample ID"
    # parts = []
    items = {}

    def __init__(self, path_to_xlsx):
        # super().__init__()
        xl = pd.ExcelFile(path_to_xlsx)
        for sheet_name in xl.sheet_names:
            print(sheet_name)
            df = xl.parse(sheet_name)
            parts = self.split(df)
            for i, p in enumerate(parts):
                try:
                    self.parse(p)
                #except AssertionError as e:
                except Exception as e:
                    print(f"part {i} skipped because of :v{e}" )
                    continue
            print(list(self.items.keys()))


    def split(self, df):
        parts = []
        last = -1
        for i, col in enumerate(df.columns):
            df[col].replace('', np.nan, inplace=True)
            if df[col][:30].isnull().all():
                parts.append(df.iloc[:, last + 1:i])
                last = i
        if i != last:
            parts.append(df.iloc[:, last + 1:i])
        return parts

    def parse(self, df):
        keys = self.get_keys(df)
        row = self.get_sample_id_row(df, keys)
        wl_row_num = self.get_wavelen_row_num(df, keys)
        wl = df.iloc[wl_row_num+1:, 0]
        for i, s_id in enumerate(row.iloc[1:]):
            s_id = str(s_id).strip()
            if len(s_id) and s_id != 'nan':
                # assert not s_id in self.items, s_id
                #s_id_str = self.find_keyphrase(keys.keys(),SAMPLE_ID)
                values = df.iloc[wl_row_num+1:, i+1]
                spectre = pd.concat([wl, values], axis=1)
                spectre = spectre.dropna(how='all')
                if s_id in self.items:
                    x = np.concatenate([spectre.to_numpy(), self.items[s_id]], axis=0)
                    self.items[s_id] = x
                else:
                    self.items[s_id] = spectre.to_numpy()

    def get_sample_id_row(self, df, keys):
        s_id_str = self.find_keyphrase(keys.keys(),SAMPLE_ID)
        n = keys[s_id_str]
        x = df.iloc[n]
        return x

    def get_wavelen_row_num(self,df,keys):
        wl_id_str = self.find_keyphrase(keys, WAVELENGTH)
        return keys[wl_id_str]

    def get_keys(self, df):
        out = {}
        # print(df.iloc[:20,0])
        for i, k in enumerate(df.iloc[:, 0]):
            if k:
                out[k] = i
            if k in WAVELENGTH:
                break
        assert self.find_keyphrase(out, SAMPLE_ID) is not None, f" Sample id({SAMPLE_ID}) not found in {out}"
        assert self.find_keyphrase(out, WAVELENGTH) is not None, f" Wavelength({WAVELENGTH}) not found in {out}"

        for wl in df.iloc[i:, 0]:
            self.wavelength.append(wl)
        return out

    def find_keyphrase(self, arr, keyphrase_synonims):
        intersection = list(set(keyphrase_synonims) & set(arr))
        if len(intersection) == 0:
            return None
        return intersection[0]

    def __len__(self):
        return len(self.items.keys())

    def __getitem__(self, n):
        keys = list(self.items.keys())
        keys.sort()
        key = keys[n]
        a = self.items[key]
        sorted = a[a[:, 0].argsort()]
        #sorted = np.sort(arr,axis=0)
        return key, sorted
