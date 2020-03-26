import pandas as pd
from utils import *
import numpy as np


def populate_harvested_area(df_main, df_harvested_area):
    for i, row in df_harvested_area.iterrows():
        area = row["Value"]
        reg = row["Region"]
        reg_col = get_region_col(reg)
        if reg_col is None:
            continue
        start_year, start_month = get_year_month(row["Start Date"])
        end_year, end_month = get_year_month(row["End Date"])
        assert start_year == end_year
        for j, row_main in df_main.iterrows():
            if row_main[reg_col] == 1 and row_main["year"] == start_year:
                df_main.iloc[j]["harvested_area"] = np.log1p(area)
    return df_main


def populate_planted_area(df_main, df_planted_area):
    for i, row in df_planted_area.iterrows():
        area = row["Value"]
        reg = row["Region"]
        reg_col = get_region_col(reg)
        if reg_col is None:
            continue
        start_year, start_month = get_year_month(row["Start Date"])
        end_year, end_month = get_year_month(row["End Date"])
        assert start_year == end_year
        for j, row_main in df_main.iterrows():
            if row_main[reg_col] == 1 and row_main["year"] == start_year:
                df_main.iloc[j]["planted_area"] = np.log1p(area)
    return df_main


def populate_canola_yield(df_main, df_canola_yield):
    for i, row in df_canola_yield.iterrows():
        yield_val = row["Value"]
        reg = row["Region"]
        reg_col = get_region_col(reg)
        if reg_col is None:
            continue
        start_year, start_month = get_year_month(row["Start Date"])
        end_year, end_month = get_year_month(row["End Date"])
        assert start_year == end_year
        for j, row_main in df_main.iterrows():
            if row_main[reg_col] == 1 and row_main["year"] == start_year:
                df_main.iloc[j]["yield"] = yield_val
    return df_main


def populate_prod(df_main, df_prod):
    for i, row in df_prod.iterrows():
        prod_val = row["Value"]
        reg = row["Region"]
        reg_col = get_region_col(reg)
        if reg_col is None:
            continue
        start_year, start_month = get_year_month(row["Start Date"])
        end_year, end_month = get_year_month(row["End Date"])
        assert start_year == end_year
        for j, row_main in df_main.iterrows():
            if row_main[reg_col] == 1 and row_main["year"] == start_year:
                df_main.iloc[j]["prod"] = np.log1p(prod_val)
    return df_main


# def populate_cropland_cover(df_main, df_cropland_cover):
#
#
# # todo populate this by personal regressors
#
#
# def populate_ndvi(df_main, df_ndvi):

