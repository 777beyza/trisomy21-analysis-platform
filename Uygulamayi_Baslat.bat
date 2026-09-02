@echo off
echo Trisomi 21 Analiz Platformu Baslatiliyor...
echo.
echo Gerekli paketler yukleniyor (bu islem ilk seferde 3-5 dakika surebilir, lutfen bekleyin)...
python -m pip install -r requirements.txt
echo.
echo Uygulama baslatiliyor, lutfen bekleyin. Tarayiciniz otomatik olarak acilacaktir.
python -m streamlit run app.py
pause
