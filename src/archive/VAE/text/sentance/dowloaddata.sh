mkdir datasets

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

tar -xf  simple-examples.tgz

mv simple-examples/data/ptb.train.txt dataset/
mv simple-examples/data/ptb.valid.txt dataset/
mv simple-examples/data/ptb.test.txt dataset/

rm -rf simple_examples
