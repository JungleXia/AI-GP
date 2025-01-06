# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *
import datetime
import numpy as np
import pandas as pd

try:
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    import joblib
except:
    import os

    print('正在安装所需库...')
    os.system('pip install scikit-learn joblib')
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    import joblib

    print('库安装完成！')

'''
示例策略仅供参考，不建议直接实盘使用。

本策略以支持向量机算法为基础，训练一个二分类（上涨/下跌）的模型，模型以历史N天数据的数据预测未来M天的涨跌与否。
特征变量为:1.收盘价/均值、2.现量/均量、3.最高价/均价、4.最低价/均价、5.现量、6.区间收益率、7.区间标准差、8.移动平均、9.RSI、10.MACD。
若没有仓位，则在每个星期一预测涨跌,并在预测结果为上涨的时候购买标的。
若已经持有仓位，则在盈利大于10%的时候止盈,在星期五涨幅小于2%的时候止盈止损。
'''


def init(context):
    # 股票标的
    context.symbol = 'SHSE.600839'
    # 历史窗口长度，N
    context.history_len = 10
    # 预测窗口长度，M
    context.forecast_len = 5
    # 训练样本长度
    context.training_len = 90  # 20天为一个交易月
    # 止盈幅度
    context.earn_rate = 0.10
    # 最小涨幅卖出幅度
    context.sell_rate = 0.02
    # 订阅行情
    subscribe(symbols=context.symbol, frequency='60s')


def on_bar(context, bars):
    bar = bars[0]
    now = context.now
    weekday = now.isoweekday()
    position = get_position()

    # 处理买卖操作
    handle_trade(context, bar, position)


def handle_trade(context, bar, position):
    if context.now.isoweekday() == 1 and context.now.hour == 9 and context.now.minute == 31 and not position:
        # 获取特征数据并进行预测
        features = clf_fit(context, context.training_len)
        features = np.array(features).reshape(1, -1)  # 确保 features 是二维数组
        prediction = context.clf.predict(features)[0]
        if prediction == 1:
            order_target_percent(symbol=context.symbol, percent=1, order_type=OrderType_Limit, position_side=PositionSide_Long, price=bar.close)

    elif position and bar.close / position[0]['vwap'] >= 1 + context.earn_rate:
        order_close_all()

    elif position and context.now.isoweekday() == 5 and bar.close / position[0][
        'vwap'] < 1 + context.sell_rate and context.now.hour == 14 and context.now.minute == 55:
        order_close_all()


def calculate_technical_indicators(recent_data):
    """
    增加更多的技术指标
    """
    # 移动平均
    recent_data['ma5'] = recent_data['close'].rolling(window=5).mean()
    recent_data['ma20'] = recent_data['close'].rolling(window=20).mean()

    # 相对强弱指数 (RSI)
    delta = recent_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    recent_data['rsi'] = 100 - (100 / (1 + rs))

    # 移动平均收敛发散指标 (MACD)
    recent_data['ema12'] = recent_data['close'].ewm(span=12).mean()
    recent_data['ema26'] = recent_data['close'].ewm(span=26).mean()
    recent_data['macd'] = recent_data['ema12'] - recent_data['ema26']

    return recent_data


def clf_fit(context, training_len):
    """
    训练支持向量机模型
    :param training_len: 训练样本长度
    """
    date_list = get_previous_n_trading_dates(exchange='SHSE', date=str(context.now.date()), n=training_len)
    start_date = date_list[0]
    end_date = date_list[-1]

    recent_data = history(context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',
                          df=True).set_index('eob')
    recent_data = calculate_technical_indicators(recent_data)

    x_train = []
    y_train = []
    for index in range(context.history_len, len(recent_data)):
        close = recent_data['close'].values
        max_x = recent_data['high'].values
        min_n = recent_data['low'].values
        volume = recent_data['volume'].values

        # 特征提取
        close_mean = close[-1] / np.mean(close)
        volume_mean = volume[-1] / np.mean(volume)
        max_mean = max_x[-1] / np.mean(max_x)
        min_mean = min_n[-1] / np.mean(min_n)
        vol = volume[-1]
        return_now = close[-1] / close[0]
        std = np.std(np.array(close), axis=0)
        ma5 = recent_data['ma5'].iloc[-1]
        ma20 = recent_data['ma20'].iloc[-1]
        rsi = recent_data['rsi'].iloc[-1]
        macd = recent_data['macd'].iloc[-1]

        # 将计算出的指标添加到训练集X
        x_train.append([close_mean, volume_mean, max_mean, min_mean, vol, return_now, std, ma5, ma20, rsi, macd])

        if index < len(recent_data) - context.forecast_len:
            y_start_date = recent_data.index[index + 1]
            y_end_date = recent_data.index[index + context.forecast_len]
            y_data = recent_data.loc[y_start_date:y_end_date, 'close']
            label = 1 if y_data.iloc[-1] > y_data.iloc[0] else 0
            y_train.append(label)

        # 最新一期的数据
        if index == len(recent_data) - 1:
            new_x_train = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std, ma5, ma20, rsi, macd]

    # 剔除最后context.forecast_len期的数据
    x_train = x_train[:-context.forecast_len]

    # 训练SVM
    context.clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
    context.clf.fit(x_train, y_train)

    # 返回最新数据
    return new_x_train


def dynamic_stop_loss(context, position, bar):
    """
    动态止损策略：基于当前价格波动调整止损幅度
    """
    avg_price = position[0]['vwap']
    price_change = (bar.close - avg_price) / avg_price

    # 动态止损：根据当前价格波动调整止损幅度
    volatility = np.std(bar.close[-20:])
    dynamic_stop_loss_rate = max(0.02, volatility * 0.5)

    # 如果跌幅大于动态止损幅度，则止损
    if price_change < -dynamic_stop_loss_rate:
        order_close_all()


def load_model():
    """
    加载保存的模型
    """
    try:
        model = joblib.load('svm_model.pkl')
        print('模型加载成功')
    except FileNotFoundError:
        model = None
        print('模型未找到，需重新训练')
    return model


def save_model(model):
    """
    保存模型
    """
    joblib.dump(model, 'svm_model.pkl')
    print('模型保存成功')


def on_order_status(context, order):
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1:
            if side == 1:
                side_effect = '开多仓'
            else:
                side_effect = '开空仓'
        else:
            if side == 1:
                side_effect = '平空仓'
            else:
                side_effect = '平多仓'
        order_type_word = '限价' if order_type == 1 else '市价'
        print(f'策略已完成委托: {side_effect}, 价格: {price}, 数量: {volume}, 委托类型: {order_type_word}')
    elif status == 4:
        print('委托失败')


if __name__ == '__main__':
    backtest_start_time = str(datetime.datetime.now() - datetime.timedelta(days=180))[:19]
    backtest_end_time = str(datetime.datetime.now())[:19]
    run(strategy_id='d07257d1-cbe8-11ef-832d-04d9f5ad60d3',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='0590a85567cb466b68df7b3101511ad3f7a644df',
        backtest_start_time=backtest_start_time,
        backtest_end_time=backtest_end_time,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
