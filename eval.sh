python test_widerface.py

cd widerface_evaluate

python evaluation.py >> result.txt

cd ..

python test_widerface.py -m ./result/mobilenet0.25_epoch_230.pth

cd widerface_evaluate

python evaluation.py >> result.txt