
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

        //DBG("win length");
        //DBG(win_length);

        if(! _hop_length){
            hop_length = floor(win_length / 4) ;
        }
        else{
            hop_length = _hop_length;
        }

        //DBG("hop length");
        //DBG(hop_length);


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

        torch::Tensor y = torch::stft(x, this->n_fft, this->hop_length, this->win_length, this->hann_win, true, "reflect", false, true, true);

        return y;
    }

    torch::Tensor batch_stft(torch::Tensor x, torch::Tensor& stftFilePhase, bool pad = true, bool return_complex = false){
        c10::IntArrayRef x_shape = x.sizes();
        
        //DBG("x_shape: ");
        //DBG(x_shape[0]);
        //DBG(x_shape[1]);
        //DBG(x_shape[2]);


        //x = x.reshape(-1, x_shape[-1])
        //x = ourReshape(x);

        //DBG("x after reshape");
        //DBG(x.sizes()[0]);
        //DBG(x.sizes()[1]);
        //DBG(x.sizes()[2]);
        
        if(pad){
            x = pad_stft_input(this->win_length, this->hop_length, x);
        }


        //DBG("x after pad");
        //DBG(x.sizes()[0]);
        //DBG(x.sizes()[1]);
        //DBG(x.sizes()[2]);
        

        torch::Tensor S = _stft(x);


        c10::IntArrayRef S_shape = S.sizes();




        if (return_complex){
            return S;
        }

        stftFilePhase = torch::angle(S);

        return torch::abs(S);


    }


    torch::Tensor _istft(torch::Tensor x, int trim_length=NULL){
        return torch::istft(x, this->n_fft, this->hop_length, this->win_length, this->hann_win, true, false, true, trim_length, false );

    }


    torch::Tensor batch_istft(torch::Tensor mag, torch::Tensor phase, int trim_length){
        torch::Tensor S = torch::polar(mag, phase);
        torch::Tensor res = _istft(S, trim_length);
        return res;

    }

};
