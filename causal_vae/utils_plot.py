import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os


#TODO: question is how is alignment plot created? 
#My guess: needs to extract masked latent encoding z and just plot those for different 

def save_A(adjacency_matrix, full_file_path):
    adjacency_matrix = adjacency_matrix.detach().squeeze().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(adjacency_matrix)
    fig.colorbar(cax)
    plt.savefig(full_file_path)
    

class VAE_Plotter: 
    def __init__(self, directory_path, plotting_vals = ["Rec", "Adjacency"], mode = "show"): 
        self.directory_path = directory_path
        self.plotting_vals = plotting_vals
        self.mode = mode
        self.__create_plot_dir()
        self._assert_plotting_vals()
    
    def __create_plot_dir(self): 
        if not os.path.isdir(self.directory_path):
            os.makedirs(self.directory_path)

    def _assert_plotting_vals(self): 
        assert self.mode in ["show", "save"]
        if isinstance(self.plotting_vals, list):
            n = len(self.plotting_vals)
            for i in range(n): 
                curr_plot_val = self.plotting_vals[i]
                print(curr_plot_val)
                assert curr_plot_val in ["Rec", "Adjacency"], "Currently only ''Rec'' and ''Adjacency'' are supported"
        else: 
            assert isinstance(self.plotting_vals, str)
            assert self.plotting_vals in ["Rec", "Adjacency"], "Currently only ''Rec'' and ''Adjacency'' are supported"
        
    def plot(self, curr_epoch, gt_img = None, img = None, adj_matrix = None):
        if "Rec" in self.plotting_vals:
            assert gt_img is not None and img is not None
            gt_img_path = f"{self.directory_path}/gt_img_{curr_epoch}.png"
            img_path = f"{self.directory_path}/rec_img_{curr_epoch}.png"
            #TODO: implement show
            save_image(gt_img, gt_img_path, normalize=True)
            save_image(img, img_path, normalize=True)
        if "Adjacency" in self.plotting_vals:
            assert adj_matrix is not None
            rec_epoch_itv = self.plotting_vals.index("Adjacency")
            adj_path = f"{self.directory_path}/adj_mat_{curr_epoch}.png"
            #TODO: implement show
            save_A(adj_matrix, adj_path)