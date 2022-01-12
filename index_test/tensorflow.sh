framework=tensorflow
if [ $# -eq 1 ]; then
	framework=$1
fi
echo $framework
cd $framework
for file in `ls *.py`; do
	echo ${file%.*} >> $framework.txt
	python $file 2>&1 | grep -E "xla time|tensorflow time|Error" >> $framework.txt
done
mv $framework.txt ..
cd -
