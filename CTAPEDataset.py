from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from math import isnan
from torch.utils.data import ConcatDataset
from glob import glob
import re
import math
from tqdm import tqdm
import os


class Sample(object):

    @staticmethod
    def create_by_row(row):
        sample_id = row['Sample ID']
        if not isinstance(sample_id, str) or len(sample_id) < 1:
            return None
        inst = Sample(sample_id)
        for name in row.index:
            t = re.match(r"^Class([0-9])?$", name)
            if t is None:
                # skip sheets without Class in title
                continue
            for num in t.groups():
                if num is None:
                    # clear substance
                    proportion = 100.0
                    cls = row['Class']
                else:
                    cls = row[f"Class{num}"]
                    proportion = row[f"W{num},%"]
                if not isinstance(cls, str) or len(cls) < 1 :
                    break
                else:
                    inst.elements.append(cls)
                    inst.proportions.append(float(proportion))

        if len(inst.elements):
            #print(inst.id, "cls == ", cls, inst.elements)
            return inst
        return None

    def __init__(self, id):
        self.elements = []
        self.proportions = []
        self.id = str(id)
        self.info = ""

    def validate(self):
        assert len(self.elements) == len(self.proportions)
        summ = sum(self.proportions)
        if  summ > 101 or summ < 99:
            raise ValueError(f"Invalid proportions {str(self)}")
        #assert sum(self.proportions) == 100 ,proportions

    def is_mixture(self):
        return len(self.elements) > 1

    def __str__(self):
        out = self.id
        for i, e in enumerate(self.elements):
            out = out + f" {e}:{self.proportions[i]} "
        return out

class SamplesLibrary(object):
    def __init__(self, path_to_xlsx,verbose = False):
        self.id2sample = {}
        xl = pd.ExcelFile(path_to_xlsx)
        pbar = tqdm(xl.sheet_names)
        skipped = []
        for sheet_name in pbar:
            pbar.set_description(sheet_name)
            #print(sheet_name)
            df = xl.parse(sheet_name)
            for index, row in df.iterrows():
                sample = Sample.create_by_row(row)
                try:
                    if sample:
                        sample.validate()
                        self.register(sample)
                except ValueError as e:
                    skipped.append(sample.id)
                    if verbose:
                        print(e)
                    continue

        xl.close()
        pbar.set_description(f"Total {len(self.get_elements())} elements in {len(self.id2sample)} samples")
        print("Skipped", skipped)


    def register(self, sample):
        if sample.id in self.id2sample:
            #print("Duplicate id",sample.id, "skipped")
            raise ValueError("Duplicate id",sample.id)
        self.id2sample[sample.id] = sample

    def has(self,sample_id):
        return sample_id in self.id2sample

    def get(self,sample_id):
        if self.has(sample_id):
            return self.id2sample[sample_id]
        return None

    def get_elements(self):
        elements = set()
        for key, s in self.id2sample.items():
            elements = elements.union(set(s.elements))
        return elements


def load_dictionary(path_to_xlsx):
    class2id = {}
    xl = pd.ExcelFile(path_to_xlsx)
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)

        for index, row in df.iterrows():
            cls = row['Class']
            if not isinstance(cls, str) or len(cls) < 1:
                continue

            sample_id = row['Sample ID']
            if not isinstance(sample_id, str) or len(sample_id) < 1:
                continue

            if cls in class2id:
                class2id[cls].add(sample_id)
            else:
                class2id[cls] = set([sample_id])

    return class2id


def load_dictionary_old(path_to_xlsx):
    class2id = {}
    xl = pd.ExcelFile(path_to_xlsx)
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)

        for index, row in df.iterrows():
            cls = row['Class']
            if not isinstance(cls, str) or len(cls) < 1:
                continue

            sample_id = row['Sample ID']
            if not isinstance(sample_id, str) or len(sample_id) < 1:
                continue

            if cls in class2id:
                class2id[cls].add(sample_id)
            else:
                class2id[cls] = set([sample_id])

    return class2id


MATERIAL_ID = ["Sample ID", "Sample #"]
WAVELENGTH = ["Wavelength (nm)", "Wavelength"]
PSF_FILE = ["PSF File", "RELAB File", "File name", "Basename", "Basename (.txt)"]


class CTAPEDataset(Dataset):
    wavelength = []
    sample_id = "Sample ID"
    # parts = []
    items = []

    def __init__(self, path_to_xlsx, elements_list, transform=None, wl_filter=(350, 900)):
        # super().__init__()
        # self.filter = self.load_filter(path_to_filter)
        if isinstance(elements_list, str):
            #self.class2id = load_dictionary(path_to_elements_list)
            elements_list = SamplesLibrary(elements_list)
        if isinstance(elements_list, SamplesLibrary) :
            self.samples_library = elements_list
        else:
            raise ValueError("Can't load list of all elements")
        self.wl_filter = wl_filter
        self.transform = transform
        xl = pd.ExcelFile(path_to_xlsx)
        self.current_sheet = path_to_xlsx.split(os.sep)[-1]
        for sheet_name in xl.sheet_names:
            print("Sheet: ",sheet_name)
            self.current_sheet += " " + sheet_name
            df = xl.parse(sheet_name)
            parts = self.split(df)
            for i, p in enumerate(parts):
                try:
                    if not p.empty:
                        self.parse(p)
                except Exception as e:
                    print(f"part {i} skipped because of : {e}")
                    continue
        xl.close()

    def load_filter(self, filename):
        filter = []
        with open(filename, "r") as file:
            for line in file:
                filter.append(line.strip().split(","))
        return filter

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
        if not self.is_wl_accepted(wl):
            raise ValueError(f"Wavelength interval must include interval [{self.wl_filter[0]}, {self.wl_filter[1]}] nm ")
        for i, s_id in enumerate(material_id_row.iloc[1:]):
            s_id = str(s_id).strip()  # + "_" + str(psf_file_name).strip()
            if self.is_id_accepted(s_id):
                # assert not s_id in self.items, s_id
                values = self.extract_values_to_first_empty_line(df.iloc[wl_row_num + 1:, i + 1])
                h = min(wl.shape[0], values.shape[0])
                if h < 2:
                    raise ValueError("Values not found",s_id)
                spectre = pd.concat([wl[:h], values[:h]], axis=1)
                #spectre = pd.concat([wl, values], axis=1)
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

    def is_id_accepted(self, s_id):
        s_id = str(s_id).strip()
        if len(s_id) == 0 or s_id == 'nan':
            return False
        if self.samples_library.has(s_id):
            #print("Found", s_id)
            return True
        #print("Not found", s_id)
        return False
        #if self.sample_id_to_class(s_id) is None:
        #    return False
        #return True

    def sample_id_to_sample(self, raw_sample_id):
        s_id = str(raw_sample_id).strip()
        return self.samples_library.get(s_id)
        #
        #for key, values in self.class2id.items():
        #    if s_id in values:
        #        return key
        #return None

    def is_wl_accepted(self, wl):
        """
            Wavelength interval must include 550 nm length
            otherwise it skipped
        """
        wl = wl.to_numpy().astype(float)
        if wl.min() <= self.wl_filter[0] and self.wl_filter[1] <= wl.max():
            return True
        return False

    def extract_values_to_first_empty_line(self, values_col):
        """
        Spectre row can contain trash, but after some empty rows
        so we truncate WL values to first empty row
        :return: PandasDataframe with non-empty rows
        """
        empty_places = np.where(pd.isnull(values_col))
        if len(empty_places[0]) > 0:
            index_of_first_empty_row = empty_places[0][0]
        else:
            # No empty lines
            index_of_first_empty_row = len(values_col)
        values = values_col.iloc[:index_of_first_empty_row]
        return values

    def spectre2array(self, spectre):
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
        spectre = self.crop_spectre(spectre)
        sample = self.sample_id_to_sample(s_id)
        if self.transform:
            spectre = self.transform(spectre)
        return sample, spectre

    def crop_spectre(self, arr):
        cropped = arr[(self.wl_filter[0] <= arr[..., 0]) & (arr[..., 0] <= self.wl_filter[1])]
        return cropped

    @property
    def classes(self):
        x = set()
        for sample_id, _ in self.items:
            sample = self.sample_id_to_sample(sample_id)
            x.update(set(sample.elements))
        return x


class MultiFileDataset(ConcatDataset):
    def __init__(self, path_to_xlsx_folder, path_to_elements_list, transform=None, wl_filter=(350, 900)):
        pattern = f"{path_to_xlsx_folder}/*.xlsx"
        files = glob(pattern)
        self._transform = transform
        datasets = []
        self.classes = set()
        self.samples_library = SamplesLibrary(path_to_elements_list)
        for f in files:
            print("File: ", f.split(os.sep)[-1])
            ds = CTAPEDataset(f, self.samples_library, transform=self._transform, wl_filter=wl_filter)
            self.classes.update(ds.classes)
            #print(ds.classes)
            datasets.append(ds)

        super().__init__(datasets)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value
        for d in self.datasets:
            d.transform = self._transform
