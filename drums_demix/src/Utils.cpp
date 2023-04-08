
#include <cmath>
#include <torch/torch.h>
#include <torch/script.h>


class Utils{
    public:
        int F;
        int T;
        int n_fft;
        int win_length;
        int hop_length;
        float power;
        bool center;
        torch::Tensor hann_win;
    Utils(int _F = 0, int _T = 0, int _n_fft = 4096, int _win_length = false, int _hop_length = false, float _power = 1.0, bool _center = true){
        n_fft = _n_fft;
        if( ! _win_length){
            win_length = _n_fft;
        }
        else {
            win_length = _win_length;
        }

        DBG("win length");
        DBG(win_length);

        if(! _hop_length){
            hop_length = floor(win_length / 4) ;
        }
        else{
            hop_length = _hop_length;
        }

        DBG("hop length");
        DBG(hop_length);


        power = _power;
        center = _center;
        F = _F;
        T = _T;
        hann_win = torch::hann_window(win_length, true, at::TensorOptions().dtype(at::kFloat).requires_grad(false));
    }

    torch::Tensor ourReshape(torch::Tensor x) {
        auto x_shape = x.sizes();
        int last = x_shape.back();
        int mult = 1;
        for (int n = 0; n < x_shape.size() - 1; n++) {
            mult = mult * x_shape[n];
        }
        torch::Tensor reshape = x.reshape({ mult,last });

        return reshape;
    }

    torch::Tensor pad_stft_input(int win_len, int hop_len, torch::Tensor x){
        auto x_shape = x.sizes();
        int last = x_shape.back();

        //pad_len = (-(x.size(-1) - self.win_length) % self.hop_length) % self.win_length
        int mod = -(last - win_len) % hop_len;
        mod = mod < 0 ? mod + hop_len : mod;
        auto pad_len = (mod) % win_len;

        torch::Tensor padx = torch::zeros({ x_shape[0], pad_len });
        torch::Tensor res = torch::cat({ x, padx }, 1);

        return res;
    }

    torch::Tensor _stft(torch::Tensor x){

        /* reflect padding, not needed if secified as a torch::stft argument

        int length = x.sizes()[0];
        int pad = length / 2;
        torch::Tensor left_pad = x.slice(0, 1, pad + 1).flip(0);
        torch::Tensor right_pad = x.slice(0, length - pad - 1, length - 1).flip(0);
        torch::Tensor x2 = torch::cat({ left_pad, x, right_pad }, 0);
        */

        torch::Tensor y = torch::stft(x, this->n_fft, this->hop_length, this->win_length, this->hann_win, true, "reflect", false, false, true);
        //torch::Tensor y = torch::stft(x2.transpose(0,1), (int) this->n_fft, (int) this->hop_length,(int) this->win_length);
        //torch::Tensor y = torch::stft(x2);

        return y;
    }

    torch::Tensor batch_stft(torch::Tensor x, torch::Tensor& stftFilePhase, bool pad = true, bool return_complex = false){
        c10::IntArrayRef x_shape = x.sizes();

        DBG("x_shape: ");
        DBG(x_shape[0]);
        DBG(x_shape[1]);
        DBG(x_shape[2]);


        //x = x.reshape(-1, x_shape[-1])
        x = ourReshape(x);

        DBG("x after reshape");
        DBG(x.sizes()[0]);
        DBG(x.sizes()[1]);
        DBG(x.sizes()[2]);
        
        if(pad){
            x = pad_stft_input(this->win_length, this->hop_length, x);
        }


        DBG("x after pad");
        DBG(x.sizes()[0]);
        DBG(x.sizes()[1]);
        DBG(x.sizes()[2]);
        

        torch::Tensor S = _stft(x);

        c10::IntArrayRef S_shape = S.sizes();

        DBG("STFT sizes: ");
        DBG(S.sizes()[0]);
        DBG(S.sizes()[1]);
        DBG(S.sizes()[2]);


        //python: S = S.reshape(x_shape[:-1] + S.shape[-2:]), utile solo per le batch

        //std::vector v(n);

        /*
        DBG(S_shape.slice(S_shape.size() - 2, 2).size());
        DBG(x_shape.slice(0, x_shape.size() - 1).size());
        DBG(n); 
        
        */

        //c10::IntArrayRef reshape_input = {x_shape.slice(0, x_shape.size() - 1), S_shape.slice(S_shape.size() - 2, 2)};

        //S = S.reshape();

        //DBG(S.sizes()[0]);
        //DBG(S.sizes()[1]);
        //DBG(S.sizes()[2]);
        //DBG(S.sizes()[3]);
        if (return_complex){
            return S;
        }

        stftFilePhase = torch::angle(S);

        return torch::abs(S);


    }

    /*
    def _istft(self, x, trim_length=None):
        return torch.istft(input=x,
                           n_fft=self.n_fft,
                           window=self.hann_window,
                           win_length=self.win_length,
                           hop_length=self.hop_length,
                           center=self.center,
                           length=trim_length
                           )
                           
                           */

    torch::Tensor _istft(torch::Tensor x, int trim_length=NULL){
        return torch::istft(x, this->n_fft, this->hop_length, this->win_length, this->hann_win, true, false, false, trim_length, false );

        //torch::stft(x, this->n_fft, this->hop_length, this->win_length, this->hann_win, true, "reflect", false, false, true);
    }

    /*    def batch_istft(self, magnitude, phase, trim_length=None):
        S = torch.polar(magnitude, phase)
        S_shape = S.size()
        S = S.reshape(-1, S_shape[-2], S_shape[-1])
        x = self._istft(S, trim_length)
        x = x.reshape(S_shape[:-2] + x.shape[-1:])
        return x*/

    torch::Tensor batch_istft(torch::Tensor mag, torch::Tensor phase, int trim_length){
        torch::Tensor S = torch::polar(mag, phase);
        torch::Tensor res = _istft(S, trim_length);
        return res;

    }

}

/*
    def __init__(self, F: int = None, T: int = None, n_fft: int = 4096, win_length: int = None, hop_length: int = None,
                 power: float = 1.0, center: bool = True, device='cpu'):

        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length
        self.hann_window = torch.hann_window(self.win_length, periodic=True).to(device)
        self.power = power
        self.center = center
        self.device = device
        self.F = F
        self.T = T


    def _stft(self, x):
        return torch.stft(input=x,
                          n_fft=self.n_fft,
                          window=self.hann_window,
                          win_length=self.win_length,
                          hop_length=self.hop_length,
                          center=self.center,
                          return_complex=True
                          )

    def batch_stft(self, x, pad: bool = True, return_complex: bool = False):
        x_shape = x.size()
        x = x.reshape(-1, x_shape[-1])
        if pad:
            x = self.pad_stft_input(x)
        S = self._stft(x)
        S = S.reshape(x_shape[:-1] + S.shape[-2:])
        if return_complex:
            return S
        return S.abs(), S.angle()

*/

/*

    def fold_unet_inputs(self, spec):
        time_dim = spec.size(-1)
        pad_len = math.ceil(time_dim / self.T) * self.T - time_dim
        padded = F.pad(spec, (0, pad_len))
        if time_dim < self.T:
            return padded
        out = torch.cat(torch.split(padded, self.T, dim=-1), dim=0)
        return out

    def unfold_unet_outputs(self, x, input_size):
        batch_size, n_frames = input_size[0], input_size[-1]
        if x.size(0) == batch_size:
            return x[..., :n_frames]
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)
        return x[..., :n_frames]

    def trim_freq_dim(self, x):
        return x[..., :self.F, :]

    def pad_freq_dim(self, x):
        padding = (self.n_fft // 2 + 1) - x.size(-2)
        x = F.pad(x, (0, 0, 0, padding))
        return x

    def pad_stft_input(self, x):
        pad_len = (-(x.size(-1) - self.win_length) % self.hop_length) % self.win_length
        return F.pad(x, (0, pad_len))


    def _istft(self, x, trim_length=None):
        return torch.istft(input=x,
                           n_fft=self.n_fft,
                           window=self.hann_window,
                           win_length=self.win_length,
                           hop_length=self.hop_length,
                           center=self.center,
                           length=trim_length
                           )

    def batch_stft(self, x, pad: bool = True, return_complex: bool = False):
        x_shape = x.size()
        x = x.reshape(-1, x_shape[-1])
        if pad:
            x = self.pad_stft_input(x)
        S = self._stft(x)
        S = S.reshape(x_shape[:-1] + S.shape[-2:])
        if return_complex:
            return S
        return S.abs(), S.angle()

    def batch_istft(self, magnitude, phase, trim_length=None):
        S = torch.polar(magnitude, phase)
        S_shape = S.size()
        S = S.reshape(-1, S_shape[-2], S_shape[-1])
        x = self._istft(S, trim_length)
        x = x.reshape(S_shape[:-2] + x.shape[-1:])
        return x
*/;
