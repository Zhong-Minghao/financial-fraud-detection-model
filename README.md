# financial fraud detection model

This is the model for predicting the financial statement fraud in Chinese stock market, my project during my intern in China's Security.

## Usage

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

Please first set the torch environment.

Due to confidentiality requrement, the /input file is empty, which should not and has a "财务违规数据表" file downloaded from the CSMAR database[CSMAR](https://data.csmar.com/) .
And also, the file link to the wind database, where you need to change the file path in xxxxxxxxxxxxxx.

```bash
result_df.to_csv(r'./result/'+stock+'xueqiu.csv')
```

To change the target stock pool, please change the [000300pool.xlsx](./000300pool.xlsx)

Then

```bash
python xueqiu_scrape.py
```


