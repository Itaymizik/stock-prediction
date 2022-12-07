# Itay Mizikov , 315541615
# preform two algorithems that predicts the next investment of each stock and calculate the profits of each day

# None of these imports are strictly required, but use of at least some is strongly encouraged
# Other imports which don't require installation can be used without consulting with course staff.
# If you feel these aren't sufficient, and you need other modules which require installation,
# you're welcome to consult with the course staff.

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance
from typing import List


class PortfolioBuilder:

    def skalar_multiple(self, list_1, list_2):    # vector multiple by coordinate
        sum1 = 0
        for i in range(len(list_1)):
            mul = list_1[i] * list_2[i]
            sum1 = sum1 + mul
        return sum1


    def vector_b (self, a, num_stocks):        # creates all vectors possible when 'a' jumps is given (aka vector "b omega")_
        e = int(a)
        data = [float(np.longdouble(i / a)) for i in range(a + 1)]
        portfolios = [p for p in itertools.product(data, repeat=num_stocks) if abs(sum(p) - 1) < 1 / (10 * e)]
        return np.asarray(portfolios)
    


    def list_x(self,t, num_stocks):                        # retun list that holds all the ratio of each day from each stock
        vector_x = []
        for j in range(0, t - 1):                          # t-1 becuase we are taking j+1 index in li
            list_betta = []                                # vector that holds the ratio of each stock in day t (price in day t / price in day t-1)
            for k in range(num_stocks):                    # k is the stock were dealing with
                a = (self.li[j + 1][k]) / (self.li[j][k])  # creates the vector x from "li" dict from above
                list_betta.append(a)
            vector_x.append(list_betta)
        return vector_x

    def s_list2(self,t, a, num_stocks):

        "holds all the S skalars with two indexes in a matrix , size of matrix : col=amount of vectors b(omega) , rows = number of days-1"

        matrix = np.zeros([t - 1, len(self.vector_b(a, num_stocks))])
        for i in range(len(self.vector_b(a, num_stocks))):
            val = 1
            for j in range(len(self.list_x(t, num_stocks))):
                val = val * self.skalar_multiple(self.vector_b(a, num_stocks)[i], self.list_x(t, num_stocks)[j])
                matrix[j][i] = val
        return matrix                # return super_list

    def uni_por_algo3(self,t, a, num_stocks):

        next_b = []
        self.b_vector = self.vector_b(a, num_stocks)      # its the b(omega) vector
        super_list = self.s_list2(t, a, num_stocks)

        for i in range(t - 1):
            sum_up = 0
            for j in range(len(self.b_vector)):
                sum_up = sum_up + (self.b_vector[j] * super_list[i][j])
            sum_dn = sum(super_list[i])
            next_b.append(sum_up / sum_dn)

        return next_b


    def expo_grad(self, t, n, num_stocks):        # n for nablla

        first_b = []
        x_vector = self.list_x(t, num_stocks)
        b_vector = []
        for i in range(num_stocks):
            first_b.append(1 / num_stocks)
        b_vector.append(first_b)
        for i in range(t - 1):
            b_num_up = []
            b_num_dn = []
            list_div = []
            for j in range(num_stocks):
                num_up = b_vector[i][j] * np.exp((n * x_vector[i][j]) / self.skalar_multiple(b_vector[i], x_vector[i]))
                b_num_up.append(num_up)
            for l in range(num_stocks):
                sum_dn = 0
                for k in range(num_stocks):
                    sum_dn = sum_dn + (b_vector[i][k] * np.exp(
                        (n * x_vector[i][k]) / self.skalar_multiple(b_vector[i], x_vector[i])))
                b_num_dn.append(sum_dn)
            for p in range(len(b_num_up)):
                list_div.append((b_num_up[p]) / (b_num_dn[p]))
            b_vector.append(list_div)
        return b_vector



    def get_daily_data(self, tickers_list: List[str],
                       start_date: date,
                       end_date: date = date.today()
                       ) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """

        try:
            self.board = web.DataReader(tickers_list, start=start_date, end=end_date, data_source='yahoo')['Adj Close']
            self.num_stocks = len(tickers_list)
            self.len_t = len(self.board)
            self.li = {}                                          # dict that holds the values of each row (aka date)
            for j in range(self.len_t):
                self.li[j] = []

            for i in range(self.len_t):                           # len of df is number of active dates between start_date and end_date
                self.li[i] = list(self.board.iloc[i, :])          # inserts the value of each stock to li by date

            return self.board

        except:
            raise ValueError



    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        a = portfolio_quantization
        list_money = [1.0]
        first_b = []                      # crates the initialize of vector b (all the vector should be the same values)
        spi = 1
        for i in range(self.num_stocks):
            first_b.append(1 / self.num_stocks)
        self.b_vector = self.uni_por_algo3(self.len_t, a, self.num_stocks)
        self.b_vector.insert(0, first_b)
        self.b_vector.pop()
        self.x_vector = self.list_x(self.len_t, self.num_stocks)
        for j in range(self.len_t - 1):
            spi = spi * self.skalar_multiple(self.b_vector[j], self.x_vector[j])
            list_money.append(spi)
        return list_money




    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """

        li = {}                                    # dict that holds the values of each row (aka date)
        for j in range(len(self.board)):
            li[j] = []
        for i in range(len(self.board)):           # len of df is number of active dates between start_date and end_date
            li[i] = list(self.board.iloc[i, :])    # inserts the value of each stock to li by date
        n = learn_rate
        list_money = [1.0]
        spi = 1
        b_vector = self.expo_grad(self.len_t, n, self.num_stocks)
        x_vector = self.list_x(self.len_t, self.num_stocks)
        for j in range(self.len_t - 1):
            spi = spi * self.skalar_multiple(b_vector[j], x_vector[j])
            list_money.append(spi)
        return list_money





if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    t = PortfolioBuilder()
    print(t.get_daily_data(["GOOG","MSFT"],date(2021,1,1),date(2021,2,1)))
    print(t.find_universal_portfolio(20))
    print(t.find_exponential_gradient_portfolio(0.5))

