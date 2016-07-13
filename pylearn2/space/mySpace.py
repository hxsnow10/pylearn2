from __init__ import Space
from theano import tensor as T
from theano.ifelse import ifelse
class DenseSquenceSpace(Space):
    '''
    发现不能用theano实现dtype=object即array(array([1,2,3]),array([2,3,4,5]))用theano只能用generic表示，然而似乎不能支持Index
    用tensor实现的话,对一个batch shape=[batch_num,*sample.shape=(sequnece_shape,...)] 只要把batch[:,0,:]用来指示seq长度
    '''
    
    '''
    #这个就是概念上的，     

class SequenceSpace(Space):
    
    def __init__(elem_space):
        self.elem_space=elem_space
    
    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return ('%s(element_space=%s)' %
                (self.__class__.__name__,
                 self.space))
    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return (type(self) == type(other) and 
                self.space == other.space)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self), self.dim, self.sparse, self.dtype))

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return self.space.get_origin()

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        a=[self.get_origin(),]*batch_size
        return np.array(a)

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[0]

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        dtype = self._clean_dtype_arg(dtype)
        origin_batch = self.get_origin_batch(batch_size, dtype)
        rval=theano.shared(origin_batch, name=name)
        return rval
    '''
    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        to_type = None

    @functools.wraps(Space._undo_format_as_impl)
    def _undo_format_as_impl(self, batch, space):
    '''


    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """
        super(IndexSequenceSpace, self)._validate_impl(is_numeric, batch)
        if is_numetic:
            if not isinstance(batch, np.ndarray) \
               and str(type(batch)) != "<type 'CudaNdarray'>":
                raise TypeError("The value of a IndexSequenceSpace batch "
                                "should be a numpy.ndarray, or CudaNdarray, "
                                "but is %s." % str(type(batch)))
            if not space._validate_impl(is_numeric, batch[0]):
                raise TypeError("subSpace error!")
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("IndexSequenceSpace batch should be a theano "
                                "Variable, got " + str(type(batch)))
            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError("IndexSequenceSpace batch should be "
                                "TensorType or CudaNdarrayType, got " +
                                str(batch.type))
            for val in get_debug_values(batch):
                self.np_validate(val)

    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, SequenceSapce):
            pass
        elif isinstance(space, Conv2DSpace):
            # check self.elem_space
            assert isinstance(self.elem_space, IndexSpace)
            assert space.shape[1]==batch[0][0].shape[0]
            sen_len=space.shape[0]
            vec_len=space.shape[1]
            # 接下来有2种方式来进行转化  1. 先截补，再进行类型转化   2. 先把每个elem类型转化，再截补，再合并
            # 多截少补 0
            
            append=lambda x: \
                ifelse(x.shape[0]>sen_len,x[:sen_len,:],subtensor(T.fmatrix(shape=(sen_len,vec_len))[:,x.shape[0]],x))
            
            rval,_=map(fn=append,sequence=batch) 
            # 最后batch(dtype=obj)转化类型为batch(dtype=float32)

        elif isinstance()
