from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from math import isnan

MATERIAL_ID = ["Sample ID", "Sample #"]
WAVELENGTH = ["Wavelength (nm)", "Wavelength"]
PSF_FILE = ["PSF File", "RELAB File", "File name", "Basename", "Basename (.txt)"]


class CTAPEDataset(Dataset):
    wavelength = []
    sample_id = "Sample ID"
    # parts = []
    items = []

    def __init__(self, path_to_xlsx):
        # super().__init__()
        xl = pd.ExcelFile(path_to_xlsx)
        for sheet_name in xl.sheet_names:
            print(sheet_name)
            df = xl.parse(sheet_name)
            parts = self.split(df)
            for i, p in enumerate(parts):
                try:
                    if not p.empty:
                        self.parse(p)
                except Exception as e:
                    print(f"part {i} skipped because of :v{e}")
                    continue
            print(self.classes())
            # print(list(self.items.keys()))
        xl.close()

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
        keywords = self.get_keywords(df)
        material_id_row = self.get_sample_id_row(df, keywords)
        # psf_file_row = self.get_ps_file_row_num(keywords)
        wl_row_num = self.get_wavelen_row_num(keywords)
        wl = self.extract_values_to_first_empty_line(df.iloc[wl_row_num + 1:, 0])
        for i, s_id in enumerate(material_id_row.iloc[1:]):
            # psf_file_name = psf_file_row.iloc[1+i]
            s_id = str(s_id).strip()  # + "_" + str(psf_file_name).strip()
            if len(s_id) and s_id != 'nan':
                # assert not s_id in self.items, s_id
                values = self.extract_values_to_first_empty_line(df.iloc[wl_row_num + 1:, i + 1])
                spectre = pd.concat([wl, values], axis=1)
                spectre = self.spectre2array(spectre)
                self.items.append([s_id, spectre])

    def get_keywords(self, df):
        out = {}
        # print(df.iloc[:20,0])
        for i, k in enumerate(df.iloc[:, 0]):
            if k:
                out[k] = i
            if k in WAVELENGTH:
                break
        for kw in [MATERIAL_ID, WAVELENGTH]:
            assert self.find_keyphrase(out, kw) is not None, f" Sample id({kw}) not found in {out}"

        for wl in df.iloc[i:, 0]:
            self.wavelength.append(wl)
        return out

    def get_sample_id_row(self, df, keys):
        s_id_str = self.find_keyphrase(keys.keys(), MATERIAL_ID)
        n = keys[s_id_str]
        x = df.iloc[n]
        return x

    def get_wavelen_row_num(self, keys):
        wl_id_str = self.find_keyphrase(keys, WAVELENGTH)
        return keys[wl_id_str]

    def get_ps_file_row(self, df, keys):
        ps_id_str = self.find_keyphrase(keys, PSF_FILE)
        n = keys[ps_id_str]
        x = df.iloc[n]
        return x

    def extract_values_to_first_empty_line(self, values_col):
        """
        Spectre row can contain trash, but after some empty rows
        so we truncate WL values to first empty row
        :return: PandasDataframe with non-empty rows
        """
        empty_places = np.where(pd.isnull(values_col))
        index_of_first_empty_row = empty_places[0][0]
        values = values_col.iloc[:index_of_first_empty_row]
        return values

    def spectre2array(self,spectre):
        spectre = spectre.dropna(how='all')
        spectre = spectre.to_numpy().astype(float)
        return spectre

    def find_keyphrase(self, arr, keyphrase_synonims):
        intersection = list(set(keyphrase_synonims) & set(arr))
        if len(intersection) == 0:
            return None
        return intersection[0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        s_id, a = self.items[n]
        a = a.astype(float)
        spectre = a[a[:, 0].argsort()]

        return s_id, spectre

    def classes(self):
        x = set()
        for s_id, _ in self.items:
            x.add(s_id)
        return x
