# Download and extract
wget https://mirror.nju.edu.cn/zsh/zsh-5.9.tar.xz
tar -xzf zsh-5.9.tar.xz
# rm zsh-5.4.2.tar.gz
cd zsh-5.9

# I will install to $HOME/local -- change it to suit your case
mkdir ~/local
# check install directory
./configure --prefix=$HOME/local
make
# all tests should pass or skip
make check
make install