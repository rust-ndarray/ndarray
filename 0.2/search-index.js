var searchIndex = {};
searchIndex['ndarray'] = {"items":[[0,"","ndarray","The `ndarray` crate provides an N-dimensional container similar to numpy’s\nndarray.",null,null],[3,"Indexes","","An iterator over the indexes of an array shape.",null,null],[3,"Si","","A slice, a description of a range of an array axis.",null,null],[3,"InnerIter","","An iterator that traverses over all dimensions but the innermost,\nand yields each inner row.",null,null],[3,"InnerIterMut","","An iterator that traverses over all dimensions but the innermost,\nand yields each inner row (mutable).",null,null],[3,"ArrayBase","","An *N*-dimensional array.",null,null],[3,"Elements","","An iterator over the elements of an array.",null,null],[3,"ElementsMut","","An iterator over the elements of an array (mutable).",null,null],[3,"Indexed","","An iterator over the indexes and elements of an array.",null,null],[3,"IndexedMut","","An iterator over the indexes and elements of an array (mutable).",null,null],[4,"ShapeError","","An error that can be produced by `.into_shape()`",null,null],[13,"IncompatibleShapes","","incompatible shapes in reshape, (from, to)",0,null],[13,"IncompatibleLayout","","incompatible layout: not contiguous",0,null],[13,"DimensionTooLarge","","Dimension too large (shape)",0,null],[5,"zeros","","Return an array filled with zeros",null,{"inputs":[{"name":"d"}],"output":{"name":"ownedarray"}}],[5,"arr0","","Return a zero-dimensional array with the element `x`.",null,{"inputs":[{"name":"a"}],"output":{"name":"array"}}],[5,"arr1","","Return a one-dimensional array with elements from `xs`.",null,null],[5,"aview0","","Return a zero-dimensional array view borrowing `x`.",null,{"inputs":[{"name":"a"}],"output":{"name":"arrayview"}}],[5,"aview1","","Return a one-dimensional array view with elements borrowing `xs`.",null,null],[5,"aview2","","Return a two-dimensional array view with elements borrowing `xs`.",null,null],[5,"aview_mut1","","Return a one-dimensional read-write array view with elements borrowing `xs`.",null,null],[5,"arr2","","Return a two-dimensional array with elements from `xs`.",null,null],[5,"arr3","","Return a three-dimensional array with elements from `xs`.",null,null],[0,"linalg","","***Deprecated: linalg is not in good shape.***",null,null],[5,"eye","ndarray::linalg","Return the identity matrix of dimension *n*.",null,{"inputs":[{"name":"ix"}],"output":{"name":"mat"}}],[5,"least_squares","","Solve *a x = b* with linear least squares approximation.",null,{"inputs":[{"name":"mat"},{"name":"col"}],"output":{"name":"col"}}],[5,"cholesky","","Factor *a = L L<sup>T</sup>*.",null,{"inputs":[{"name":"mat"}],"output":{"name":"mat"}}],[5,"subst_fw","","Solve *L x = b* where *L* is a lower triangular matrix.",null,{"inputs":[{"name":"mat"},{"name":"col"}],"output":{"name":"col"}}],[5,"subst_bw","","Solve *U x = b* where *U* is an upper triangular matrix.",null,{"inputs":[{"name":"mat"},{"name":"col"}],"output":{"name":"col"}}],[6,"Col","","Column vector.",null,null],[6,"Mat","","Rectangular matrix.",null,null],[8,"Ring","","Trait union for a ring with 1.",null,null],[8,"Field","","Trait union for a field.",null,null],[8,"ComplexField","","A real or complex number.",null,null],[11,"conjugate","","",1,{"inputs":[{"name":"complexfield"}],"output":{"name":"self"}}],[10,"sqrt_real","","",1,{"inputs":[{"name":"complexfield"}],"output":{"name":"self"}}],[11,"is_complex","","",1,{"inputs":[{"name":"complexfield"}],"output":{"name":"bool"}}],[11,"conjugate","num::complex","",2,{"inputs":[{"name":"complex"}],"output":{"name":"complex"}}],[11,"sqrt_real","","",2,{"inputs":[{"name":"complex"}],"output":{"name":"complex"}}],[11,"is_complex","","",2,{"inputs":[{"name":"complex"}],"output":{"name":"bool"}}],[11,"index","ndarray","",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"elem"}}],[11,"index_mut","","",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"elem"}}],[11,"eq","","Return `true` if the array shapes and all elements of `self` and\n`rhs` are equal. Return `false` otherwise.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"bool"}}],[11,"from_iter","","",3,{"inputs":[{"name":"arraybase"},{"name":"i"}],"output":{"name":"arraybase"}}],[11,"hash","","",3,{"inputs":[{"name":"arraybase"},{"name":"h"}],"output":null}],[11,"encode","","",3,{"inputs":[{"name":"arraybase"},{"name":"e"}],"output":{"name":"result"}}],[11,"decode","","",3,{"inputs":[{"name":"arraybase"},{"name":"e"}],"output":{"name":"result"}}],[11,"fmt","","",3,{"inputs":[{"name":"arraybase"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",3,{"inputs":[{"name":"arraybase"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",3,{"inputs":[{"name":"arraybase"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",3,{"inputs":[{"name":"arraybase"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",3,{"inputs":[{"name":"arraybase"},{"name":"formatter"}],"output":{"name":"result"}}],[0,"blas","","Experimental BLAS (Basic Linear Algebra Subprograms) integration",null,null],[3,"BlasArrayViewMut","ndarray::blas","***Requires `features = \"rblas\"`***",null,null],[8,"AsBlas","","Convert an array into a blas friendly wrapper.",null,null],[10,"blas_checked","","Return an array view implementing Vector (1D) or Matrix (2D)\ntraits.",4,{"inputs":[{"name":"asblas"}],"output":{"name":"result"}}],[11,"blas","","Equivalent to `.blas_checked().unwrap()`",4,{"inputs":[{"name":"asblas"}],"output":{"name":"blasarrayviewmut"}}],[11,"blas_checked","ndarray","",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"result"}}],[11,"len","ndarray::blas","",5,{"inputs":[{"name":"blasarrayviewmut"}],"output":{"name":"c_int"}}],[11,"as_ptr","","",5,null],[11,"as_mut_ptr","","",5,null],[11,"inc","","",5,{"inputs":[{"name":"blasarrayviewmut"}],"output":{"name":"c_int"}}],[11,"rows","","",5,{"inputs":[{"name":"blasarrayviewmut"}],"output":{"name":"c_int"}}],[11,"cols","","",5,{"inputs":[{"name":"blasarrayviewmut"}],"output":{"name":"c_int"}}],[11,"lead_dim","","",5,{"inputs":[{"name":"blasarrayviewmut"}],"output":{"name":"c_int"}}],[11,"as_ptr","","",5,null],[11,"as_mut_ptr","","",5,null],[11,"ndim","collections::vec","",6,{"inputs":[{"name":"vec"}],"output":{"name":"usize"}}],[11,"slice","","",6,null],[11,"slice_mut","","",6,null],[11,"remove_axis","","",6,{"inputs":[{"name":"vec"},{"name":"usize"}],"output":{"name":"vec"}}],[11,"clone","ndarray","",7,{"inputs":[{"name":"indexes"}],"output":{"name":"indexes"}}],[11,"new","","Create an iterator over the array shape `dim`.",7,{"inputs":[{"name":"indexes"},{"name":"d"}],"output":{"name":"indexes"}}],[11,"next","","",7,{"inputs":[{"name":"indexes"}],"output":{"name":"option"}}],[11,"size_hint","","",7,null],[11,"clone","","",8,{"inputs":[{"name":"elements"}],"output":{"name":"elements"}}],[11,"next","","",8,{"inputs":[{"name":"elements"}],"output":{"name":"option"}}],[11,"size_hint","","",8,null],[11,"next_back","","",8,{"inputs":[{"name":"elements"}],"output":{"name":"option"}}],[11,"next","","",9,{"inputs":[{"name":"indexed"}],"output":{"name":"option"}}],[11,"size_hint","","",9,null],[11,"next","","",10,{"inputs":[{"name":"elementsmut"}],"output":{"name":"option"}}],[11,"size_hint","","",10,null],[11,"next_back","","",10,{"inputs":[{"name":"elementsmut"}],"output":{"name":"option"}}],[11,"next","","",11,{"inputs":[{"name":"indexedmut"}],"output":{"name":"option"}}],[11,"size_hint","","",11,null],[11,"next","","",12,{"inputs":[{"name":"inneriter"}],"output":{"name":"option"}}],[11,"next","","",13,{"inputs":[{"name":"inneritermut"}],"output":{"name":"option"}}],[11,"fmt","","",14,{"inputs":[{"name":"si"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"hash","","",14,null],[11,"eq","","",14,{"inputs":[{"name":"si"},{"name":"si"}],"output":{"name":"bool"}}],[11,"ne","","",14,{"inputs":[{"name":"si"},{"name":"si"}],"output":{"name":"bool"}}],[11,"clone","","",14,{"inputs":[{"name":"si"}],"output":{"name":"si"}}],[11,"from","","",14,{"inputs":[{"name":"si"},{"name":"range"}],"output":{"name":"si"}}],[11,"from","","",14,{"inputs":[{"name":"si"},{"name":"rangefrom"}],"output":{"name":"si"}}],[11,"from","","",14,{"inputs":[{"name":"si"},{"name":"rangeto"}],"output":{"name":"si"}}],[11,"from","","",14,{"inputs":[{"name":"si"},{"name":"rangefull"}],"output":{"name":"si"}}],[11,"step","","",14,{"inputs":[{"name":"si"},{"name":"ixs"}],"output":{"name":"self"}}],[11,"fmt","","",0,{"inputs":[{"name":"shapeerror"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",0,{"inputs":[{"name":"shapeerror"}],"output":{"name":"shapeerror"}}],[11,"description","","",0,{"inputs":[{"name":"shapeerror"}],"output":{"name":"str"}}],[11,"fmt","","",0,{"inputs":[{"name":"shapeerror"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"add","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"sub","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"mul","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"div","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"rem","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"bitand","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"bitor","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"bitxor","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"shl","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"shr","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"neg","","Perform an elementwise negation of `self` and return the result.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"self"}}],[11,"not","","Perform an elementwise unary not of `self` and return the result.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"self"}}],[11,"add_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"sub_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"mul_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"div_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"rem_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"bitand_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"bitor_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"bitxor_assign","","",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[6,"Ix","","Array index type",null,null],[6,"Ixs","","Array index type (signed)",null,null],[6,"Array","","Array where the data is reference counted and copy on write, it\ncan act as both an owner as the data as well as a lightweight view.",null,null],[6,"OwnedArray","","Array where the data is owned uniquely.",null,null],[6,"ArrayView","","A lightweight array view.",null,null],[6,"ArrayViewMut","","A lightweight read-write array view.",null,null],[17,"S","","Slice value for the full range of an axis.",null,null],[8,"Dimension","","Trait for the shape and index types of arrays.",null,null],[16,"SliceArg","","`SliceArg` is the type which is used to specify slicing for this\ndimension.",15,null],[10,"ndim","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"usize"}}],[11,"slice","","",15,null],[11,"slice_mut","","",15,null],[11,"size","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"usize"}}],[11,"default_strides","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"self"}}],[11,"first_index","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"option"}}],[11,"next_for","","Iteration -- Use self as size, and return next index after `index`\nor None if there are no more.",15,{"inputs":[{"name":"dimension"},{"name":"self"}],"output":{"name":"option"}}],[11,"stride_offset","","Return stride offset for index.",15,{"inputs":[{"name":"dimension"},{"name":"self"},{"name":"self"}],"output":{"name":"isize"}}],[11,"stride_offset_checked","","Return stride offset for this dimension and index.",15,{"inputs":[{"name":"dimension"},{"name":"self"},{"name":"self"}],"output":{"name":"option"}}],[11,"do_slices","","Modify dimension, strides and return data pointer offset",15,{"inputs":[{"name":"dimension"},{"name":"self"},{"name":"self"},{"name":"slicearg"}],"output":{"name":"isize"}}],[8,"RemoveAxis","","Helper trait to define a larger-than relation for array shapes:\nremoving one axis from *Self* gives smaller dimension *Smaller*.",null,null],[16,"Smaller","","",16,null],[10,"remove_axis","","",16,{"inputs":[{"name":"removeaxis"},{"name":"usize"}],"output":{"name":"smaller"}}],[8,"Data","","Array’s inner representation.",null,null],[16,"Elem","","",17,null],[10,"slice","","",17,null],[8,"DataMut","","Array’s writable inner representation.",null,null],[10,"slice_mut","","",18,null],[11,"ensure_unique","","",18,{"inputs":[{"name":"datamut"},{"name":"arraybase"}],"output":null}],[8,"DataClone","","Clone an Array’s storage.",null,null],[10,"clone_with_ptr","","Unsafe because, `ptr` must point inside the current storage.",19,null],[8,"DataOwned","","Array representation that is a unique or shared owner of its data.",null,null],[10,"new","","",20,{"inputs":[{"name":"dataowned"},{"name":"vec"}],"output":{"name":"self"}}],[10,"into_shared","","",20,{"inputs":[{"name":"dataowned"}],"output":{"name":"rc"}}],[8,"DataShared","","Array representation that is a lightweight view.",null,null],[8,"Initializer","","Slice or fixed-size array used for array initialization",null,null],[16,"Elem","","",21,null],[10,"as_init_slice","","",21,null],[11,"is_fixed_size","","",21,{"inputs":[{"name":"initializer"}],"output":{"name":"bool"}}],[8,"FixedInitializer","","Fixed-size array used for array initialization",null,null],[10,"len","","",22,{"inputs":[{"name":"fixedinitializer"}],"output":{"name":"usize"}}],[11,"slice","alloc::rc","",23,null],[11,"slice_mut","","",23,null],[11,"ensure_unique","","",23,{"inputs":[{"name":"rc"},{"name":"arraybase"}],"output":null}],[11,"clone_with_ptr","","",23,null],[11,"slice","collections::vec","",6,null],[11,"slice_mut","","",6,null],[11,"clone_with_ptr","","",6,null],[11,"new","","",6,{"inputs":[{"name":"vec"},{"name":"vec"}],"output":{"name":"self"}}],[11,"into_shared","","",6,{"inputs":[{"name":"vec"}],"output":{"name":"rc"}}],[11,"new","alloc::rc","",23,{"inputs":[{"name":"rc"},{"name":"vec"}],"output":{"name":"self"}}],[11,"into_shared","","",23,{"inputs":[{"name":"rc"}],"output":{"name":"rc"}}],[11,"clone","ndarray","",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"from_vec","","Create a one-dimensional array from a vector (no allocation needed).",3,{"inputs":[{"name":"arraybase"},{"name":"vec"}],"output":{"name":"arraybase"}}],[11,"from_iter","","Create a one-dimensional array from an iterable.",3,{"inputs":[{"name":"arraybase"},{"name":"i"}],"output":{"name":"arraybase"}}],[11,"linspace","","Create a one-dimensional array from inclusive interval\n`[start, end]` with `n` elements. `F` must be a floating point type.",3,{"inputs":[{"name":"arraybase"},{"name":"f"},{"name":"f"},{"name":"usize"}],"output":{"name":"arraybase"}}],[11,"range","","Create a one-dimensional array from interval `[start, end)`",3,{"inputs":[{"name":"arraybase"},{"name":"f32"},{"name":"f32"}],"output":{"name":"arraybase"}}],[11,"from_elem","","Construct an array with copies of `elem`, dimension `dim`.",3,{"inputs":[{"name":"arraybase"},{"name":"d"},{"name":"a"}],"output":{"name":"arraybase"}}],[11,"zeros","","Construct an array with zeros, dimension `dim`.",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"arraybase"}}],[11,"default","","Construct an array with default values, dimension `dim`.",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"arraybase"}}],[11,"from_vec_dim","","Create an array from a vector (with no allocation needed).",3,{"inputs":[{"name":"arraybase"},{"name":"d"},{"name":"vec"}],"output":{"name":"arraybase"}}],[11,"len","","Return the total number of elements in the Array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"usize"}}],[11,"dim","","Return the shape of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"d"}}],[11,"shape","","Return the shape of the array as a slice.",3,null],[11,"strides","","Return the strides of the array",3,null],[11,"view","","Return a read-only view of the array",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"arrayview"}}],[11,"view_mut","","Return a read-write view of the array",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"arrayviewmut"}}],[11,"to_owned","","Return an uniquely owned copy of the array",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"ownedarray"}}],[11,"to_shared","","Return a shared ownership (copy on write) array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"array"}}],[11,"into_shared","","Turn the array into a shared ownership (copy on write) array,\nwithout any copying.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"array"}}],[11,"iter","","Return an iterator of references to the elements of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"elements"}}],[11,"indexed_iter","","Return an iterator of references to the elements of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"indexed"}}],[11,"iter_mut","","Return an iterator of mutable references to the elements of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"elementsmut"}}],[11,"indexed_iter_mut","","Return an iterator of indexes and mutable references to the elements of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"indexedmut"}}],[11,"slice","","Return a sliced array.",3,{"inputs":[{"name":"arraybase"},{"name":"slicearg"}],"output":{"name":"self"}}],[11,"islice","","Slice the array’s view in place.",3,{"inputs":[{"name":"arraybase"},{"name":"slicearg"}],"output":null}],[11,"slice_iter","","Return an iterator over a sliced view.",3,{"inputs":[{"name":"arraybase"},{"name":"slicearg"}],"output":{"name":"elements"}}],[11,"slice_mut","","Return a sliced read-write view of the array.",3,{"inputs":[{"name":"arraybase"},{"name":"slicearg"}],"output":{"name":"arrayviewmut"}}],[11,"slice_iter_mut","","***Deprecated: use `.slice_mut()`***",3,{"inputs":[{"name":"arraybase"},{"name":"slicearg"}],"output":{"name":"elementsmut"}}],[11,"get","","Return a reference to the element at `index`, or return `None`\nif the index is out of bounds.",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"option"}}],[11,"at","","***Deprecated: use .get(i)***",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"option"}}],[11,"get_mut","","Return a mutable reference to the element at `index`, or return `None`\nif the index is out of bounds.",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"option"}}],[11,"at_mut","","***Deprecated: use .get_mut(i)***",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"option"}}],[11,"uget","","Perform *unchecked* array indexing.",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"a"}}],[11,"uchk_at","","***Deprecated: use `.uget()`***",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"a"}}],[11,"uget_mut","","Perform *unchecked* array indexing.",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"a"}}],[11,"uchk_at_mut","","***Deprecated: use `.uget_mut()`***",3,{"inputs":[{"name":"arraybase"},{"name":"d"}],"output":{"name":"a"}}],[11,"swap_axes","","Swap axes `ax` and `bx`.",3,{"inputs":[{"name":"arraybase"},{"name":"usize"},{"name":"usize"}],"output":null}],[11,"subview","","Along `axis`, select the subview `index` and return an\narray with that axis removed.",3,{"inputs":[{"name":"arraybase"},{"name":"usize"},{"name":"ix"}],"output":{"name":"arraybase"}}],[11,"isubview","","Collapse dimension `axis` into length one,\nand select the subview of `index` along that axis.",3,{"inputs":[{"name":"arraybase"},{"name":"usize"},{"name":"ix"}],"output":null}],[11,"subview_mut","","Along `axis`, select the subview `index` and return a read-write view\nwith the axis removed.",3,{"inputs":[{"name":"arraybase"},{"name":"usize"},{"name":"ix"}],"output":{"name":"arrayviewmut"}}],[11,"sub_iter_mut","","***Deprecated: use `.subview_mut()`***",3,{"inputs":[{"name":"arraybase"},{"name":"usize"},{"name":"ix"}],"output":{"name":"elementsmut"}}],[11,"inner_iter","","Return an iterator that traverses over all dimensions but the innermost,\nand yields each inner row.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"inneriter"}}],[11,"inner_iter_mut","","Return an iterator that traverses over all dimensions but the innermost,\nand yields each inner row.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"inneritermut"}}],[11,"diag_iter","","Return an iterator over the diagonal elements of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"elements"}}],[11,"diag","","Return the diagonal as a one-dimensional array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"arraybase"}}],[11,"diag_mut","","Return a read-write view over the diagonal elements of the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"arrayviewmut"}}],[11,"diag_iter_mut","","***Deprecated: use `.diag_mut()`***",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"elementsmut"}}],[11,"is_standard_layout","","Return `true` if the array data is laid out in contiguous “C order” in\nmemory (where the last index is the most rapidly varying).",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"bool"}}],[11,"as_slice","","Return the array’s data as a slice, if it is contiguous and\nthe element order corresponds to the memory order. Return `None` otherwise.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"option"}}],[11,"as_slice_mut","","Return the array’s data as a slice, if it is contiguous and\nthe element order corresponds to the memory order. Return `None` otherwise.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"option"}}],[11,"reshape","","Transform the array into `shape`; any shape with the same number of\nelements is accepted.",3,{"inputs":[{"name":"arraybase"},{"name":"e"}],"output":{"name":"arraybase"}}],[11,"into_shape","","Transform the array into `shape`; any shape with the same number of\nelements is accepted, but the source array or view must be\ncontiguous, otherwise we cannot rearrange the dimension.",3,{"inputs":[{"name":"arraybase"},{"name":"e"}],"output":{"name":"result"}}],[11,"broadcast","","Act like a larger size and/or shape array by *broadcasting*\ninto a larger shape, if possible.",3,{"inputs":[{"name":"arraybase"},{"name":"e"}],"output":{"name":"option"}}],[11,"broadcast_iter","","***Deprecated: Use `.broadcast()` instead.***",3,{"inputs":[{"name":"arraybase"},{"name":"e"}],"output":{"name":"option"}}],[11,"raw_data","","Return a slice of the array’s backing data in memory order.",3,null],[11,"raw_data_mut","","Return a mutable slice of the array’s backing data in memory order.",3,null],[11,"assign","","Perform an elementwise assigment to `self` from `rhs`.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"assign_scalar","","Perform an elementwise assigment to `self` from scalar `x`.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"zip_mut_with","","Traverse two arrays in unspecified order, in lock step,\ncalling the closure `f` on each element pair.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"},{"name":"f"}],"output":null}],[11,"fold","","Traverse the array elements in order and apply a fold,\nreturning the resulting value.",3,{"inputs":[{"name":"arraybase"},{"name":"b"},{"name":"f"}],"output":{"name":"b"}}],[11,"map","","Apply `f` elementwise and return a new array with\nthe results.",3,{"inputs":[{"name":"arraybase"},{"name":"f"}],"output":{"name":"ownedarray"}}],[11,"sum","","Return sum along `axis`.",3,{"inputs":[{"name":"arraybase"},{"name":"usize"}],"output":{"name":"ownedarray"}}],[11,"scalar_sum","","Return the sum of all elements in the array.",3,{"inputs":[{"name":"arraybase"}],"output":{"name":"a"}}],[11,"mean","","Return mean along `axis`.",3,{"inputs":[{"name":"arraybase"},{"name":"usize"}],"output":{"name":"ownedarray"}}],[11,"allclose","","Return `true` if the arrays' elementwise differences are all within\nthe given absolute tolerance.<br>\nReturn `false` otherwise, or if the shapes disagree.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"},{"name":"a"}],"output":{"name":"bool"}}],[11,"row_iter","","Return an iterator over the elements of row `index`.",3,{"inputs":[{"name":"arraybase"},{"name":"ix"}],"output":{"name":"elements"}}],[11,"col_iter","","Return an iterator over the elements of column `index`.",3,{"inputs":[{"name":"arraybase"},{"name":"ix"}],"output":{"name":"elements"}}],[11,"mat_mul","","Perform matrix multiplication of rectangular arrays `self` and `rhs`.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"array"}}],[11,"mat_mul_col","","Perform the matrix multiplication of the rectangular array `self` and\ncolumn vector `rhs`.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":{"name":"array"}}],[11,"iadd","","Perform elementwise\naddition\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"iadd_scalar","","Perform elementwise\naddition\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"isub","","Perform elementwise\nsubtraction\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"isub_scalar","","Perform elementwise\nsubtraction\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"imul","","Perform elementwise\nmultiplication\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"imul_scalar","","Perform elementwise\nmultiplication\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"idiv","","Perform elementwise\ndivision\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"idiv_scalar","","Perform elementwise\ndivision\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"irem","","Perform elementwise\nremainder\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"irem_scalar","","Perform elementwise\nremainder\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"ibitand","","Perform elementwise\nbit and\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"ibitand_scalar","","Perform elementwise\nbit and\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"ibitor","","Perform elementwise\nbit or\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"ibitor_scalar","","Perform elementwise\nbit or\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"ibitxor","","Perform elementwise\nbit xor\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"ibitxor_scalar","","Perform elementwise\nbit xor\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"ishl","","Perform elementwise\nleft shift\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"ishl_scalar","","Perform elementwise\nleft shift\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"ishr","","Perform elementwise\nright shift\n between `self` and `rhs`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"arraybase"}],"output":null}],[11,"ishr_scalar","","Perform elementwise\nright shift\n between `self` and the scalar `x`,\n *in place*.",3,{"inputs":[{"name":"arraybase"},{"name":"a"}],"output":null}],[11,"ineg","","Perform an elementwise negation of `self`, *in place*.",3,{"inputs":[{"name":"arraybase"}],"output":null}],[11,"inot","","Perform an elementwise unary not of `self`, *in place*.",3,{"inputs":[{"name":"arraybase"}],"output":null}],[11,"clone","","",9,{"inputs":[{"name":"indexed"}],"output":{"name":"indexed"}}],[14,"s!","","Slice argument constructor.",null,null],[11,"into_iter","","",24,{"inputs":[{"name":"arrayview"}],"output":{"name":"intoiter"}}],[11,"into_iter","","",25,{"inputs":[{"name":"arrayviewmut"}],"output":{"name":"intoiter"}}],[11,"slice","","",15,null],[11,"slice_mut","","",15,null],[11,"size","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"usize"}}],[11,"default_strides","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"self"}}],[11,"first_index","","",15,{"inputs":[{"name":"dimension"}],"output":{"name":"option"}}],[11,"next_for","","Iteration -- Use self as size, and return next index after `index`\nor None if there are no more.",15,{"inputs":[{"name":"dimension"},{"name":"self"}],"output":{"name":"option"}}],[11,"stride_offset","","Return stride offset for index.",15,{"inputs":[{"name":"dimension"},{"name":"self"},{"name":"self"}],"output":{"name":"isize"}}],[11,"stride_offset_checked","","Return stride offset for this dimension and index.",15,{"inputs":[{"name":"dimension"},{"name":"self"},{"name":"self"}],"output":{"name":"option"}}],[11,"do_slices","","Modify dimension, strides and return data pointer offset",15,{"inputs":[{"name":"dimension"},{"name":"self"},{"name":"self"},{"name":"slicearg"}],"output":{"name":"isize"}}],[11,"ndim","","",26,{"inputs":[{"name":"ix"}],"output":{"name":"usize"}}],[11,"size","","",26,{"inputs":[{"name":"ix"}],"output":{"name":"usize"}}],[11,"default_strides","","",26,{"inputs":[{"name":"ix"}],"output":{"name":"self"}}],[11,"first_index","","",26,{"inputs":[{"name":"ix"}],"output":{"name":"option"}}],[11,"next_for","","",26,{"inputs":[{"name":"ix"},{"name":"ix"}],"output":{"name":"option"}}],[11,"stride_offset","","Self is an index, return the stride offset",26,{"inputs":[{"name":"ix"},{"name":"ix"},{"name":"ix"}],"output":{"name":"isize"}}],[11,"stride_offset_checked","","Return stride offset for this dimension and index.",26,{"inputs":[{"name":"ix"},{"name":"ix"},{"name":"ix"}],"output":{"name":"option"}}],[11,"remove_axis","","",26,null]],"paths":[[4,"ShapeError"],[8,"ComplexField"],[3,"Complex"],[3,"ArrayBase"],[8,"AsBlas"],[3,"BlasArrayViewMut"],[3,"Vec"],[3,"Indexes"],[3,"Elements"],[3,"Indexed"],[3,"ElementsMut"],[3,"IndexedMut"],[3,"InnerIter"],[3,"InnerIterMut"],[3,"Si"],[8,"Dimension"],[8,"RemoveAxis"],[8,"Data"],[8,"DataMut"],[8,"DataClone"],[8,"DataOwned"],[8,"Initializer"],[8,"FixedInitializer"],[3,"Rc"],[6,"ArrayView"],[6,"ArrayViewMut"],[6,"Ix"]]};
initSearch(searchIndex);