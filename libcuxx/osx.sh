# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LIBCUXX_INSTALL_PATH=$SCRIPTPATH/build_local

if [ -z "$VANAHEIMR_INSTALL_PATH" ]
then
	export VANAHEIMR_INSTALL_PATH=$SCRIPTPATH/../vanaheimr/build_local 
fi

export PATH=$LIBCUXX_INSTALL_PATH/bin:$VANAHEIMR_INSTALL_PATH/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LIBCUXX_INSTALL_PATH/lib

#if [ -n "$LLVM_INSTALL_PATH" ]
#then
#	export PATH=$LLVM_INSTALL_PATH/bin:$PATH
#fi

export CXX=clang++
export CC=clang
export CPP=clang

