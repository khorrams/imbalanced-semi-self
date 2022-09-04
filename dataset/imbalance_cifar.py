import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
import json
import torch
import PIL
import sys
sys.path.append("stylegan2ada")
import dnnlib
import legacy


class ImbalanceCIFAR10SYNS(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False, add_syns_data=False,
                 stylegan_path=None, xflip=False,):
        super(ImbalanceCIFAR10SYNS, self).__init__(root, train, transform, target_transform, download)

        self.imb_factor = imb_factor
        self.xflip = xflip
        self.rand_number = rand_number

        if add_syns_data:
            assert stylegan_path

        self.classes = set(self.targets)
        self.img_max = len(self.data) / self.cls_num

        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

        if add_syns_data:
            # syns file name
            fname = f"syns_imf{self.imb_factor}"
            if not self.syns_data_exist():
                # generate syns imgs
                gap_img_num_per_cls = [int(self.img_max - i) for i in img_num_list]
                syns_data = self.gen_syns_data(stylegan_path, gap_img_num_per_cls)
                # save syns imgs
                with open(os.path.join(self.root, f"{fname}.json"), "w") as f:
                    json.dump({"labels": syns_data}, f)
            else:
                # load syns imgs
                with open(os.path.join(self.root, f"{fname}.json")) as f:
                    syns_data = json.load(f)["labels"]
            self.load_syns_imgs(syns_data)
            print(f"Total of {len(syns_data)} synthetic data added.")

        if self.xflip:
            raise NotImplementedError

    def load_syns_imgs(self, syns_data):
        print("Loading synthetic data..")
        data, targets = [], []
        for fn, target in syns_data:
            img_name = os.path.join(self.root, fn)
            img = PIL.Image.open(img_name).convert('RGB')
            img = np.asarray(img)
            data.append(img)
            targets.append(target)

        data = np.asarray(data)
        self.data = np.concatenate((self.data, data), axis=0)
        self.targets.extend(targets)

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
        # load pretrained generator, discriminator, and encoder
        with dnnlib.util.open_url(stylegan_path) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
            G.eval()
            G.requires_grad = False
        assert G.c_dim

        syns_list = []
        for class_idx, count in zip(self.classes, gap_img_num_per_cls):
            # make outdir
            print(f"Generating {count} images for class {class_idx}..")
            cur_dir = f"imf{self.imb_factor}/syns_{class_idx:05d}"
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
                z = torch.from_numpy(np.random.RandomState(self.rand_number).randn(cur_batch_size, G.z_dim)).to(device)
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

    def syns_data_exist(self):  # TODO count the image number inside each folder.
        """

        :return:
        """
        exists = True
        for class_idx in self.classes:
            cur_dir = f"imf{self.imb_factor}/syns_{class_idx:05d}"
            cur_path = os.path.join(self.root, cur_dir)
            if not os.path.exists(cur_path):
                exists = False
        return exists

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = self.img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(self.img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(self.img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(self.img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        print(f"Building CIFAR10-LT with IMF {self.imb_factor} (Total of {sum(img_num_per_cls)} images)")
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR100SYNS(ImbalanceCIFAR10):
    pass


class ImbalanceCIFAR100(ImbalanceCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class SemiSupervisedImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    unlabel_size_factor = 5

    def __init__(self, root, imb_type='exp', imb_factor=0.01, unlabel_imb_factor=1,
                 rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(SemiSupervisedImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        # unlabeled
        self.unlabeled_data = os.path.join(root, 'ti_80M_selected.pickle')  # selected data from 80M-TI
        self.unlabeled_pseudo = os.path.join(root, 'pseudo_labeled_cifar.pickle')  # pseudo-label using model trained on imbalanced data
        self.imb_factor = imb_factor
        self.unlabel_imb_factor = unlabel_imb_factor
        self.num_per_cls_dict = dict()

        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        img_num_list_unlabeled = self.get_img_num_per_cls_unlabeled(self.cls_num, img_num_list, unlabel_imb_factor)
        self.gen_imbalanced_data(img_num_list, img_num_list_unlabeled)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def get_img_num_per_cls_unlabeled(self, cls_num, labeled_img_num_list, imb_factor):
        img_unlabeled_total = np.sum(labeled_img_num_list) * self.unlabel_size_factor
        img_first_min = img_unlabeled_total // cls_num
        img_num_per_cls_unlabel = []
        for cls_idx in range(cls_num):
            num = img_first_min * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls_unlabel.append(int(num))
        factor = img_unlabeled_total / np.sum(img_num_per_cls_unlabel)
        img_num_per_cls_unlabel = [int(num * factor) for num in img_num_per_cls_unlabel]
        print(f"Unlabeled est total:\t{img_unlabeled_total}\n"
              f"After processing:\t{np.sum(img_num_per_cls_unlabel)},\t{img_num_per_cls_unlabel}")
        return img_num_per_cls_unlabel

    def gen_imbalanced_data(self, img_num_per_cls, img_num_per_cls_unlabeled):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        print(f"Labeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

        # unlabeled dataset
        print("Loading unlabeled data from %s" % self.unlabeled_data)
        with open(self.unlabeled_data, 'rb') as f:
            aux = pickle.load(f)
        aux_data = aux['data']
        aux_truth = aux['extrapolated_targets']
        print("Loading pseudo labels from %s" % self.unlabeled_pseudo)
        with open(self.unlabeled_pseudo, 'rb') as f:
            aux_targets = pickle.load(f)
        aux_targets = aux_targets['extrapolated_targets']

        for the_class, the_img_num in zip(classes, img_num_per_cls_unlabeled):
            # ground truth is only used to select samples
            idx = np.where(aux_truth == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(aux_data[selec_idx, ...])
            # append pseudo-label
            new_targets.extend(aux_targets[selec_idx])
            for pseudo_class in aux_targets[selec_idx]:
                self.num_per_cls_dict[pseudo_class] += 1
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'
        print(f"Unlabeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = SemiSupervisedImbalanceCIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
