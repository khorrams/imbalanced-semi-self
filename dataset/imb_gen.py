import torchvision
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import PIL
import json


class LtDataset(torch.utils.data.Dataset):
    def __init__(self, root="data/cifar10", fname="dataset.json", imf=100, reverse=False,
                 add_syns_data=False, stylegan_path=None,
                 xflip=False, random_seed=0):
        """
            Loads data given a path (root_dir) and preprocess them (transforms, blur)
        :param root_dir:
        :param transform:
        :param blur:
        """
        self.root = root
        self.fname = fname
        self.imf = imf
        self.reverse = reverse
        self.xflip = xflip
        self.random_seed = random_seed

        if add_syns_data:
            assert stylegan_path

        print(f"\nLoading filenames from '{fname}' file...")

        with open(os.path.join(self.root, self.fname)) as f:
            data_ = json.load(f)

        if data_["labels"]:
            data = np.array(data_["labels"], dtype=object)
        else:
            raise RuntimeError

        self.class_dist = self.get_class_dist(data)
        self.classes = set(self.class_dist)
        self.cls_num = len(self.classes)
        self.img_max = self.class_dist[max(self.class_dist, key=self.class_dist.get)]

        lt_img_num_per_cls = self.get_lt_img_num_per_cls()
        self.lt_data = self.gen_imbalanced_data(data, lt_img_num_per_cls)

        if add_syns_data:
            # syns file name
            fname = f"syns_imf{self.imf}"
            if self.reverse:
                fname += "_reverse"

            if not self.syns_data_exist():
                # generate syns imgs
                gap_img_num_per_cls = [self.img_max - i for i in lt_img_num_per_cls]
                syns_data = self.gen_syns_data(stylegan_path, gap_img_num_per_cls)
                # save syns imgs
                with open(os.path.join(self.root, f"{fname}.json"), "w") as f:
                    json.dump({"labels": syns_data}, f)
            else:
                # load syns imgs
                with open(os.path.join(self.root, f"{fname}.json")) as f:
                    syns_data = json.load(f)["labels"]
            # add syns data
            self.lt_data.extend(syns_data)

        if self.xflip:
            raise NotImplementedError

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname, label = self.filenames[idx]
        img_name = os.path.join(self.root, fname)
        image = PIL.Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.lt_data)

    def syns_data_exist(self):  # TODO count the image number inside each folder.
        """

        :return:
        """
        exists = True
        for class_idx in self.classes:
            cur_dir = f"syns_{class_idx:05d}"
            cur_path = os.path.join(self.root, cur_dir)
            if not os.path.exists(cur_path):
                exists = False
        return exists

    def gen_syns_data(self,
                      stylegan_path,
                      gap_img_num_per_cls,
                      batch_size=16,
                      truncation_psi=0.9,
                      noise_mode="const",
                      device="cuda"):
        """

        :param stylegan_path:
        :param gap_img_num_per_cls:
        :param batch_size:
        :param truncation_psi:
        :param noise_mode:
        :param device:
        :return:
        """
        import dnnlib
        import legacy

        # load pretrained generator, discriminator, and encoder
        with dnnlib.util.open_url(stylegan_path) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
            G.eval()
            G.requires_grad = False
        assert G.c_dim

        syns_list = []
        for class_idx, count in zip(self.classes, gap_img_num_per_cls):
            # make outdir
            cur_dir = f"syns_{class_idx:05d}"
            cur_path = os.path.join(self.root, cur_dir)
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)

            rem = count
            idx = 0
            while rem > 0:
                cur_batch_size = batch_size if rem >= batch_size else rem
                rem -= cur_batch_size
                # label
                label = torch.zeros([cur_batch_size, G.c_dim], device=device)
                label[:, class_idx] = 1
                # noise
                z = torch.from_numpy(np.random.RandomState(self.random_seed).randn(cur_batch_size, G.z_dim)).to(device)
                # generate
                imgs = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # save images
                for img in imgs:
                    file_name = f"img_syn{idx:08d}.png"
                    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(os.path.join(self.root, cur_dir, file_name))
                    syns_list.append([f"{cur_dir}/{file_name}", class_idx])
                    idx += 1
        return syns_list

    @ staticmethod
    def get_class_dist(data):
        """

        :param data:
        :return:
        """
        class_dist = dict()
        for d in data:
            if d[1] in class_dist:
                class_dist[d[1]] += 1
            else:
                class_dist[d[1]] = 1
        return class_dist

    def get_lt_img_num_per_cls(self):
        """

        :return:
        """
        img_num_per_cls = []
        for cls_idx in self.classes:
            if self.reverse:
                num = self.img_max * (1 / self.imf ** ((self.cls_num - 1 - cls_idx) / (self.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
            else:
                num = self.img_max * (1 / self.imf ** (cls_idx / (self.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, ds, lt_img_num_per_cls):
        """

        :param ds:
        :param lt_img_num_per_cls:
        :return:
        """
        lt_data = []
        for i, count in zip(self.classes, lt_img_num_per_cls):
            idx = np.where(ds[:, 1] == i)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:count]
            lt_data.append(ds[selec_idx, ...])
        return np.vstack(lt_data).tolist()


# if __name__ == "__main__":
#     import sys
#     sys.path.append("stylegan2ada")
#     d = LtDataset(add_syns_data=True, stylegan_path="./snaps/network-snapshot-028224.pkl")
#     print(d.__len__())