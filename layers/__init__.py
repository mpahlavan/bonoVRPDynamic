from ._mha         import _MHA_V2 as MultiHeadAttention
#from _bigbird_claude import BigBirdMHAttention as BBMultiHeadAttention
from ._bigbird     import BigBirdMultiHeadAttention as MultiHeadAttention
from ._transformer import TransformerEncoder, TransformerEncoderLayer
from ._loss        import reinforce_loss
