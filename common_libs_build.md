### gcc 编译
gcc下载链接：https://ftp.gnu.org/gnu/
gcc官网安装教程：https://gcc.gnu.org/wiki/InstallingGCC

编译gcc7.5.0过程
```bash
% gcc_version=7.5.0
% wget --no-check-certificate https://ftp.gnu.org/gnu/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.gz
% wget --no-check-certificate https://ftp.gnu.org/gnu/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.gz.sig
% wget --no-check-certificate https://ftp.gnu.org/gnu/gnu-keyring.gpg
% signature_invalid=`gpg --verify --no-default-keyring --keyring ./gnu-keyring.gpg gcc-${gcc_version}.tar.gz.sig`
% if [ $signature_invalid ]; then echo "Invalid signature" ; exit 1 ; fi
% tar -xzvf gcc-${gcc_version}.tar.gz
% cd gcc-${gcc_version}
% ./contrib/download_prerequisites
% cd ..
% mkdir gcc-${gcc_version}-build
% cd gcc-${gcc_version}-build
% $PWD/../gcc-${gcc_version}/configure --prefix=$HOME/toolchains --enable-languages=c,c++ --disable-multilib
% make -j$(nproc)
% make install
```
在编译gcc的时候遇到问题 https://www.cnblogs.com/haiyang21/p/10828134.html
```bash
../../gcc-aarch64-sve-acle-branch/gcc/hwint.h:62:5: error: #error "Unable to find a suitable type for HOST_WIDE_INT"
    #error "Unable to find a suitable type for HOST_WIDE_INT"
     ^
In file included from ../../gcc-aarch64-sve-acle-branch/gcc/hash-table.h:243:0,
                 from ../../gcc-aarch64-sve-acle-branch/gcc/coretypes.h:441,
                 from ../../gcc-aarch64-sve-acle-branch/gcc/c/c-lang.c:23:
../../gcc-aarch64-sve-acle-branch/gcc/statistics.h:25:2: error: #error GATHER_STATISTICS must be defined
 #error GATHER_STATISTICS must be defined
```
解决方案：export CPLUS_INCLUDE_PATH=
##### 相关知识点：gcc、libstdc++、libc之间的关系


### llvm 编译
clang官网安装方法链接： https://clang.llvm.org/get_started.html
llvm官网安装方法链接： https://llvm.org/docs/GettingStarted.html#requirements
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout remotes/origin/release/9.x %如果想要切换到某个版本时使用
mkdir build 
cd build
cmake -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$HOME/toolchains/lib64 -L$HOME/toolchains/lib64" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;openmp;polly" -DCMAKE_INSTALL_PREFIX="$HOME/toolchains" -G Ninja ../llvm
%% rpath以及L指定编译时使用的gcc版本以及libstdc++版本， 也可以通过设置CMAKE_INSTALL_PREFIX将其安装路径设置为与gcc一致来完成。
ninja
sudo ninja install
```

### pytorch源码安装

### pytorch lazytensor编译
torch要求clang的版本不高于8


### torch-mlir编译
```
cmake -GNinja -Bbuild \
  -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$HOME/toolchains/lib64 -L$HOME/toolchains/lib64" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=`pwd` \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  external/llvm-project/llvm
```
可能遇到的问题
- cpu-mkl找不到openmp库，因此在编译llvm clang时，需要在cmake命令中显示指定-DLLVM_ENABLE_PROJECTS="clang;openmp"。
- 显示libstdc++版本低于5.1,手动多个gcc版本后，发现8.5.0满足条件
- 更新cmake版本到3.18以上

### anoconda如何多版本cuda和tensorflow pytorch等库共存
- 从https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/ 清华镜像下载anaconda软件包，5-2-0版本对应
- 安装时，直接bash 安装包.sh，安装过程中一般直接按enter就行，如果想要换个安装路径，可以在安装过程中输入。
- anaconda基本命令
    - conda upgrade conda
    - conda install python=3.8
    - conda env list 罗列所有的虚拟环境
    - conda create -n test python=3.7 创建虚拟环境
    - 报CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'时，运行source ~/anaconda3/etc/profile.d/conda.sh
    - conda remove -n test --all //删除test环境
    - conda remove -n test numpy //删除test中的numpy
    - conda env export > environment.yaml // 导出当前环境的包信息
    - conda env create -f environment.yaml // 用配置文件创建新的虚拟环境
    - conda install requests
    - conda remove requests 或者 conda uninstall requests
    - 如何设置下载源（pip和conda）:https://blog.csdn.net/program_developer/article/details/79677557
    - conda search cudatoolkit
    - conda search cudnn

### tmux
https://www.ruanyifeng.com/blog/2019/10/tmux.html
https://www.scutmath.com/tmux_session_save_restore.html
http://louiszhai.github.io/2017/09/30/tmux/#%E6%96%B0%E5%BB%BA%E4%BC%9A%E8%AF%9D
- 安装： sudo apt-get install tmux
- 退出：Ctrl + d或者exit退出tmux
- 新建会话：tmux new -s tensorflow-2.7
- 脱离当前会话：tmux detach【尽量不要用C+d，会把会话直接删除】
- 显示所有会话：tmux ls或者tmux list-session
- 接入会话：tmux attach -t <session-name>
- 杀死会话：tmux kill-session -t 0/<session-name>
- 会话切换：tmux switch -t 0/<session-name>
- 重命名会话：tmux rename-session -t 0 <new-name>
- 保存会话：
    1. 克隆tmux插件管理器：git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
    2. 在.tmux.conf中添加如下命令，tmux的快捷键前缀默认是ctrl+b，这里可以通过在.tmux.conf中添加命令set -g prefix C-a进行修改
    ```
    # List of plugins
    set -g @plugin 'tmux-plugins/tpm'
    set -g @plugin 'tmux-plugins/tmux-sensible'

    # Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)
    run '~/.tmux/plugins/tpm/tpm'
    ```
    3. 更新配置文件：tmux source-file ~/.tmux.conf
    4. 按快捷键：Ctrl+a + I安装插件（这里可能需要等几秒中，待出现环境已配置等字样后按esc键退出）
    5. ctrl+a ctrl+s保存会话 //在底部状态栏会有提示
    6. ctrl+a ctrl+r恢复会话
    
### gpu驱动版本，驱动版本和cuda版本，以及向下兼容问题

- cuda版本要求gpu驱动版本： https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- cuda toolkit 安装： https://developer.nvidia.com/cuda-downloads?target_os=Linux
- cudnn安装：https://developer.nvidia.com/rdp/cudnn-archive 下载所需版本cuDNN Library for Linux (x86_64)压缩包就行，将include中的.h文件拷贝到cuda目录下的include文件夹下，并且把lib64目录下的lib*文件拷贝到cuda的lib64文件夹下。

### pytorch pip二进制安装

    https://pytorch.org/get-started/locally/
   
    - conda create -n pytorch
    - conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch //[此命令会自动安装cuda 11.3.1]
    
### jax 二进制安装
   
    https://github.com/google/jax
    https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
    
    - conda create -n jax pip
    - /home/local/SENSETIME/jianglijuan/anaconda3/envs/jax/bin/pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
    
### tensorflow 二进制安装
   
    页面https://www.tensorflow.org/install/source?hl=zh-cn 提供了tensorflow版本和cuda版本的对应关系， 在pypy页面部分tensorflow也提供了cuda版本以及python版本的限制。
    
    - conda create -n tensorflow pip
    - pip install tensorflow

    不同cuda版本和库版本共存的问题，可以暂时通过tmux不同会话中设置不同的PATH和LD_LIBRARY_PATH来简单绕过。
