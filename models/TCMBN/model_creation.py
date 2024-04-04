from .transformer.Models import Transformer

def create_model(opt):
    model = Transformer(
        num_types=opt.num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        b_comps=opt.ber_comps,
        dropout=opt.dropout,
    )
    return model