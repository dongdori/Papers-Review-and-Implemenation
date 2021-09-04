class CycleGANDataset(Dataset):
    def __init__(self, s1_root, s2_root, transform = None):
        self.s1_img_paths = list(map(lambda x: os.path.join(s1_root, x), os.listdir(s1_root)))
        self.s2_img_paths = list(map(lambda x: os.path.join(s2_root, x), os.listdir(s2_root)))
        self.transforms = transforms.Compose(transform)
        
    def __len__(self):
        return max(len(self.s1_img_paths), len(self.s2_img_paths))
    
    def __getitem__(self, index):
        s1_item = Image.open(self.s1_img_paths[index % len(self.s1_img_paths)])
        s2_item = Image.open(self.s2_img_paths[index % len(self.s2_img_paths)])
        s1_item = self.transform(s1_item)
        s2_item = self.transform(s2_item)
        return {'s1' : s1_item, 's2' : s2_item}
