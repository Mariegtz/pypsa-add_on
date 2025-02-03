# -*- coding: utf-8 -*-
import cartopy.crs as ccrs
import pandas as pd
# pd.set_option('display.max_rows', None)
import geopandas as gpd
import matplotlib.pyplot as plt
import pypsa
from pypsa.descriptors import get_switchable_as_dense as as_dense
from shapely.geometry import Point

scen = "BZ2"  # BZ1/BZ2/BZ4
solver = "gurobi"

o = pypsa.Network("grid_w_constr_01_12_no_nuclear_diffCO2.nc")

# add offshore buses baltic

o.add("Bus", "off_baltic_DE", x=13, y=55, v_nom=220)
o.add("Bus", "off_baltic_DK", x=13, y=55, v_nom=220)
o.add("Bus", "off_baltic_PL", x=13, y=55, v_nom=220)
o.add("Bus", "off_baltic_SE", x=13, y=55, v_nom=220)

# add offshore buses north
o.add("Bus", "off_north_BE", x=6, y=54, v_nom=220)
o.add("Bus", "off_north_DK", x=6, y=54, v_nom=220)
o.add("Bus", "off_north_DE", x=6, y=54, v_nom=220)
o.add("Bus", "off_north_FR", x=6, y=54, v_nom=220)
o.add("Bus", "off_north_GB", x=6, y=54, v_nom=220)
o.add("Bus", "off_north_NL", x=6, y=54, v_nom=220)
o.add("Bus", "off_north_NO", x=6, y=54, v_nom=220)

# add lines to offshore areas
# Build lines to OBZ North Sea
OBZ_link_data = {
    "bus0": ["DE1 11", "GB0 0", "GB3 0", "FR1 0", "NL1 0", "BE1 0", "DK1 0", "NO2 0", "DE1 109", "DK2 0", "PL1 0",
             "SE2 0"],
    "bus1": ["off_north_DE", "off_north_GB", "off_north_GB", "off_north_FR", "off_north_NL", "off_north_BE",
             "off_north_DK", "off_north_NO", "off_baltic_DE", "off_baltic_DK", "off_baltic_PL", "off_baltic_SE"],
    "carrier": ["DC", "DC", "DC", "DC", "DC", "DC", "DC", "DC", "DC", "DC", "DC", "DC"],
    # "p_nom": [2400, 3000, 3000, 3000, 2400, 600, 2400, 3000,969,608,1418,2006],
    "p_nom": [2880, 3600, 3600, 3600, 2880, 720, 2880, 3600, 1163, 730, 1702, 2407],  # 20
    # "p_nom": [3120, 3900, 3900, 3900, 3120, 780, 3120, 3900,1260,790,1845,2608],#30
    "p_min_pu": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
}

# Accounting to new bus
o.generators.loc["BE1 0 offwind-ac", "bus"] = "off_north_BE"
o.generators.loc["BE1 0 offwind-dc", "bus"] = "off_north_BE"
o.generators.loc["DE1 4 offwind-ac", "bus"] = "off_north_DE"
o.generators.loc["DE1 77 offwind-ac", "bus"] = "off_north_DE"
o.generators.loc["DE1 77 offwind-dc", "bus"] = "off_north_DE"
o.generators.loc["DE1 92 offwind-ac", "bus"] = "off_north_DE"
o.generators.loc["FR1 0 offwind-dc", "bus"] = "off_north_FR"
o.generators.loc["FR1 1 offwind-ac", "bus"] = "off_north_FR"
o.generators.loc["FR1 0 offwind-ac", "bus"] = "off_north_FR"
o.generators.loc["DK1 0 offwind-ac", "bus"] = "off_north_DK"
o.generators.loc["DK1 0 offwind-dc", "bus"] = "off_north_DK"

o.generators.loc["GB0 0 offwind-ac", "bus"] = "off_north_GB"
o.generators.loc["GB0 0 offwind-dc", "bus"] = "off_north_GB"
o.generators.loc["GB3 0 offwind-ac", "bus"] = "off_north_GB"
o.generators.loc["GB3 0 offwind-dc", "bus"] = "off_north_GB"

o.generators.loc["NL1 0 offwind-ac", "bus"] = "off_north_NL"
o.generators.loc["NL1 0 offwind-dc", "bus"] = "off_north_NL"
o.generators.loc["NO2 0 offwind-ac", "bus"] = "off_north_NO"
o.generators.loc["NO2 0 offwind-dc", "bus"] = "off_north_NO"

# Baltic
o.generators.loc["DE1 109 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 19 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 29 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 35 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 51 offwind-ac", "bus"] = "off_baltic_DE"

o.generators.loc["DE1 51 offwind-dc", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 65 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 65 offwind-dc", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 69 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 82 offwind-ac", "bus"] = "off_baltic_DE"

o.generators.loc["DE1 86 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 9 offwind-ac", "bus"] = "off_baltic_DE"
o.generators.loc["DE1 9 offwind-dc", "bus"] = "off_baltic_DE"

o.generators.loc["DK2 0 offwind-ac", "bus"] = "off_baltic_DK"
o.generators.loc["DK2 0 offwind-dc", "bus"] = "off_baltic_DK"
o.generators.loc["PL1 0 offwind-ac", "bus"] = "off_baltic_PL"
o.generators.loc["PL1 0 offwind-ac", "bus"] = "off_baltic_PL"
o.generators.loc["SE2 0 offwind-ac", "bus"] = "off_baltic_SE"
o.generators.loc["SE2 0 offwind-dc", "bus"] = "off_baltic_SE"

# off_bus_baltic = o.generators[o.generators.bus == "off_baltic"]
# off_bus_north = o.generators[o.generators.bus == "off_north"]

# o.mremove("Bus", ["off_baltic"])
# o.mremove("Bus", ["off_north"])

df_obz_links = pd.DataFrame(OBZ_link_data)
o.import_components_from_dataframe(df_obz_links, "Link")

# Fix expansion
o.generators["p_nom_extendable"] = False
o.storage_units["p_nom_extendable"] = False
o.lines["s_nom_extendable"] = False
o.links["p_nom_extendable"] = False
o.links["p_nom_opt"] = o.links["p_nom"]
o.links.p_max_pu = 0.7

n = o.copy()  # for redispatch model
m = o.copy()  # for market model

# Fix ausbau
# Assign p_nom_opt values from o to n and m
n.generators["p_nom"] = o.generators["p_nom_opt"]
n.storage_units["p_nom"] = o.storage_units["p_nom_opt"]
n.lines["s_nom"] = o.lines["s_nom_opt"]
n.links["p_nom"] = o.links["p_nom_opt"]  # dann werden die OBZ null
n.links.p_max_pu = 0.7

# Set expandable attribute to False
n.generators["p_nom_extendable"] = False
n.storage_units["p_nom_extendable"] = False
n.lines["s_nom_extendable"] = False
n.links["p_nom_extendable"] = False

m.generators["p_nom"] = o.generators["p_nom_opt"]
m.storage_units["p_nom"] = o.storage_units["p_nom_opt"]
m.lines["s_nom"] = o.lines["s_nom_opt"]
m.links["p_nom"] = o.links["p_nom_opt"]  # dann werden die OBZ null
m.links.p_max_pu = 0.7

# Set expandable attribute to False
m.generators["p_nom_extendable"] = False
m.storage_units["p_nom_extendable"] = False
m.lines["s_nom_extendable"] = False
m.links["p_nom_extendable"] = False

#######################################
# build market model, with bidding zones
######################################

# read in shapes
nuts_3_eu = gpd.read_file("NUTS_RG_10M_2021_4326.geojson")
# nuts_1_de = nuts_3_eu.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
nuts = nuts_3_eu.query("LEVL_CODE ==1")
obz_shapes = gpd.read_file("regions_offshore_elec_s_128.geojson")

# create one Point for every bus in one column
m.buses["Point"] = [Point(xy) for xy in zip(m.buses.x, m.buses.y)]
m.buses


def order_bus_to_state(point):
    for y in nuts["geometry"].tolist():
        if point.within(y):
            value = nuts.loc[nuts["geometry"] == y, "NUTS_ID"].iloc[0]
            return value


def order_bus_to_obz(point):
    for y in obz_shapes["geometry"].tolist():
        if point.within(y):
            value = obz_shapes.loc[obz_shapes["geometry"] == y, "name"].iloc[0]
            return value


# map the points
states = m.buses["Point"].map(lambda x: order_bus_to_state(x))
off_states = m.buses["Point"].map(lambda x: order_bus_to_obz(x))
# convert to df
df_states = states.to_frame()
df_off_states = off_states.to_frame()

# drop none rows
df_states_no_none = df_states.dropna()
df_off_states_no_none = df_off_states.dropna()
df_states_filtered = pd.concat([df_states_no_none, df_off_states_no_none])
states = df_states_filtered.squeeze()


# create zones depending on the scenario

def create_scenarios():
    if scen == "BZ2":
        lookup_dict = {"DEF": "DEII1", "DE6": "DEII1", "DE9": "DEII1", "DE3": "DEII1", "DE4": "DEII1",
                       "DE8": "DEII1", "DED": "DEII1", "DEE": "DEII1", "DEG": "DEII1", "DEA": "DEII2",
                       "DEB": "DEII2", "DEC": "DEII2", "DE1": "DEII2", "DE2": "DEII2", "DE7": "DEII2"}
    elif scen == "BZ1":
        lookup_dict = {"DEF": "DEI1", "DE6": "DEI1", "DE9": "DEI1", "DE3": "DEI1", "DE4": "DEI1",
                       "DE8": "DEI1", "DED": "DEI1", "DEE": "DEI1", "DEG": "DEI1", "DEA": "DEI1",
                       "DEB": "DEI1", "DEC": "DEI1", "DE1": "DEI1", "DE2": "DEI1", "DE7": "DEI1"}
    elif scen == "BZ4":
        lookup_dict = {"DEF": "DEIV1", "DE6": "DEIV1", "DE9": "DEIV1", "DE3": "DEIV2", "DE4": "DEIV2",
                       "DE8": "DEIV2", "DED": "DEIV2", "DEE": "DEIV2", "DEG": "DEIV2", "DEA": "DEIV3",
                       "DEB": "DEIV3", "DEC": "DEIV3", "DE1": "DEIV4", "DE2": "DEIV4", "DE7": "DEIV4"}
    return lookup_dict


# assign unassigned nodes manually to zone
if scen == "BZ2":

    states.loc[states.index == "off_baltic_DE"] = 'DEII1'
    states.loc[states.index == "off_north_DK"] = 'DK0'
    states.loc[states.index == "off_baltic_DK"] = 'DK0'
    states.loc[states.index == "off_baltic_PL"] = 'PL7'
    states.loc[states.index == "off_baltic_SE"] = 'SE3'
    states.loc[states.index == "off_north_BE"] = 'BE2'
    states.loc[states.index == "off_north_DE"] = 'DEII1'
    states.loc[states.index == "off_north_FR"] = 'FRK'
    states.loc[states.index == "off_north_GB"] = 'UKF'
    states.loc[states.index == "off_north_NL"] = 'NL2'
    states.loc[states.index == "off_north_NO"] = 'NO0'

    states.loc[states.index == 'DK2 0'] = 'DK0'  # yes
    states.loc[states.index == 'DK2 0 H2'] = 'DK0'  # yes
    states.loc[states.index == 'DK2 0 battery'] = 'DK0'  # yes
    states.loc[states.index == 'DE1 65'] = 'DEII1'  # yes
    states.loc[states.index == 'DE1 65 H2'] = 'DEII1'  # yes
    states.loc[states.index == 'DE1 65 battery'] = 'DEII1'  # yes


elif scen == "BZ1":

    states.loc[states.index == "off_baltic_DE"] = 'DEI1'
    states.loc[states.index == "off_north_DK"] = 'DK0'
    states.loc[states.index == "off_baltic_DK"] = 'DK0'
    states.loc[states.index == "off_baltic_PL"] = 'PL7'
    states.loc[states.index == "off_baltic_SE"] = 'SE3'
    states.loc[states.index == "off_north_BE"] = 'BE2'
    states.loc[states.index == "off_north_DE"] = 'DEI1'
    states.loc[states.index == "off_north_FR"] = 'FRK'
    states.loc[states.index == "off_north_GB"] = 'UKF'
    states.loc[states.index == "off_north_NL"] = 'NL2'
    states.loc[states.index == "off_north_NO"] = 'NO0'

    states.loc[states.index == 'DK2 0'] = 'DK0'  # yes
    states.loc[states.index == 'DK2 0 H2'] = 'DK0'  # yes
    states.loc[states.index == 'DK2 0 battery'] = 'DK0'  # yes
    states.loc[states.index == 'DE1 65'] = 'DEI1'  # yes
    states.loc[states.index == 'DE1 65 H2'] = 'DEI1'  # yes
    states.loc[states.index == 'DE1 65 battery'] = 'DEI1'  # yes


elif scen == "BZ4":

    states.loc[states.index == "off_baltic_DE"] = 'DEIV2'
    states.loc[states.index == "off_north_DK"] = 'DK0'
    states.loc[states.index == "off_baltic_DK"] = 'DK0'
    states.loc[states.index == "off_baltic_PL"] = 'PL7'
    states.loc[states.index == "off_baltic_SE"] = 'SE3'
    states.loc[states.index == "off_north_BE"] = 'BE2'
    states.loc[states.index == "off_north_DE"] = 'DEIV1'
    states.loc[states.index == "off_north_FR"] = 'FRK'
    states.loc[states.index == "off_north_GB"] = 'UKF'
    states.loc[states.index == "off_north_NL"] = 'NL2'
    states.loc[states.index == "off_north_NO"] = 'NO0'

    states.loc[states.index == 'DK2 0'] = 'DK0'  # yes
    states.loc[states.index == 'DK2 0 H2'] = 'DK0'  # yes
    states.loc[states.index == 'DK2 0 battery'] = 'DK0'  # yes
    states.loc[states.index == 'DE1 65'] = 'DEIV2'  # yes
    states.loc[states.index == 'DE1 65 H2'] = 'DEIV2'  # yes
    states.loc[states.index == 'DE1 65 battery'] = 'DEIV2'  # yes

zones = states.replace(create_scenarios())

indices_with_none = zones.index[zones.isna()].tolist()

for c in m.iterate_components(m.one_port_components):
    c.df.bus = c.df.bus.map(zones)

for c in m.iterate_components(m.branch_components):
    c.df.bus0 = c.df.bus0.map(zones)
    c.df.bus1 = c.df.bus1.map(zones)
    internal = c.df.bus0 == c.df.bus1
    m.mremove(c.name, c.df.loc[internal].index)

m.mremove("Bus", m.buses.index)
if scen == "BZ1":

    name = ['AT3', 'BE2', 'CH0', 'CZ0', 'DEI', 'DK0', 'FRB', 'FRK', 'UKF',
            'UKN', 'LU0', 'NL2', 'NO0', 'PL7', 'SE3']
    longitude = [13.39, 4.61, 8.21, 15.25, 10.66, 9.78, 1.11, 5.03, -1.98, -6.09, 6.13, 5.55, 8.62, 19.05, 16.17]
    latitude = [47.55, 50.81, 46.85, 49.83, 51.62, 55.95, 48.2, 45.5, 53.33, 54.66, 49.67, 52.37, 60.99, 51.57, 61.24]

elif scen == "BZ2":

    name = ['AT3', 'BE2', 'CH0', 'CZ0', 'DEII2', 'DEII1', 'DK0', 'FRB', 'FRK',
            'UKF', 'UKN', 'LU0', 'NL2', 'NO0', 'PL7', 'SE3', 'DK2']
    longitude = [13.39, 4.61, 8.21, 15.25, 10.54, 10.64, 9.78, 1.11, 5.03, -1.98, -6.09, 6.13, 5.55, 8.62, 19.05, 16.17]
    latitude = [47.55, 50.81, 46.85, 49.83, 48.9, 52.47, 55.95, 48.2, 45.5, 53.33, 54.66, 49.67, 52.37, 60.99, 51.57,
                61.24]

elif scen == "BZ4":

    name = ['AT3', 'BE2', 'CH0', 'CZ0', 'DEIV4', 'DEIV3', 'DEIV2', 'DEIV1',
            'DK0', 'FRB', 'FRK', 'UKF', 'UKN', 'LU0', 'NL2', 'NO0', 'PL7',
            'SE3']
    longitude = [13.39, 4.61, 8.21, 15.25, 10.98, 7.55, 12.77, 9.33, 9.78, 1.11, 5.03, -1.98, -6.09, 6.13, 5.55, 8.62,
                 19.05, 16.17]
    latitude = [47.55, 50.81, 46.85, 49.83, 48.8, 50.36, 52.52, 53.03, 55.95, 48.2, 45.5, 53.33, 54.66, 49.67, 52.37,
                60.99, 51.57, 61.24]

m.madd("Bus", zones.unique(), x=longitude, y=latitude)

# solve market model:
m.generators["p_nom_extendable"] = False
m.generators["p_nom"] = o.generators["p_nom_opt"]
m.optimize.create_model()

m.optimize.solve_model(
    solver_name='gurobi'
)

# m.export_to_netcdf('Solved_market_model.nc')
if scen == "BZ1":

    # m.export_to_csv_folder("../Ergebnisse/BZ1_noOBZ_markt_model_optimised_short")
    m.export_to_netcdf("../Ergebnisse/BZ1_noOBZ_markt_model_optimised.nc")

elif scen == "BZ2":

    # m.export_to_csv_folder("../Ergebnisse/BZ2_noOBZ_markt_model_optimised_short")
    m.export_to_netcdf("New_line_cap_20_BZ2_noOBZ_markt_model_optimised.nc")

elif scen == "BZ4":

    # m.export_to_csv_folder("../Ergebnisse/BZ4_noOBZ_markt_model_optimised_short")
    m.export_to_netcdf("../Ergebnisse/BZ4_noOBZ_markt_model_optimised.nc")

