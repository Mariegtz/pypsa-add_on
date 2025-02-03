import pypsa
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import plotly.graph_objects as go
import matplotlib.lines as mlines
import pandas as pd
import matplotlib
import matplotlib.ticker as ticker
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
matplotlib.use('Agg')

output_dir = "results/figures_marie"
n = pypsa.Network("results/postnetworks/base_s_10_lvopt___2040.nc")
n45 = pypsa.Network("results/postnetworks/base_s_10_lvopt___2045.nc")
network = pypsa.Network("results/postnetworks/base_s_10_lvopt___2040.nc")
#plt.ioff()

#check installed capacities
generators_2040 = pd.concat([n.generators.p_nom_opt, n45.generators.p_nom_opt], axis = 1)
generators_2040.rename(columns={"A": "2040", "B": "2045"})

def plot_dispatch(n, time="2013-07"):
    p_by_carrier = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum().div(1e3)

    if not n.storage_units.empty:
        sto = n.storage_units_t.p.T.groupby(n.storage_units.carrier).sum().T.div(1e3)
        p_by_carrier = pd.concat([p_by_carrier, sto], axis=1)

    fig, ax = plt.subplots(figsize=(6, 3))

    color = p_by_carrier.columns.map(n.carriers.color)

    p_by_carrier.where(p_by_carrier > 0).loc[time].plot.area(
        ax=ax,
        linewidth=0,
        color=color,
    )

    charge = p_by_carrier.where(p_by_carrier < 0).dropna(how="all", axis=1).loc[time]

    if not charge.empty:
        charge.plot.area(
            ax=ax,
            linewidth=0,
            color=charge.columns.map(n.carriers.color),
        )

    n.loads_t.p_set.sum(axis=1).loc[time].div(1e3).plot(ax=ax, c="k")

    plt.legend(loc=(1.05, 0))
    ax.set_ylabel("GW")
    ax.set_ylim(-200, 200)
plot_dispatch(n)

import cartopy.feature as cfeature

def load_distribution(network):

    fig, ax = plt.subplots(
        1, 1, subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(8, 8)
    )
    load_distribution = (
        network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()
    )
    load_distribution = load_distribution[~load_distribution.index.str.contains("EU")]
    network.plot(bus_sizes=1e-5 * load_distribution, ax=ax, title="Load distribution")
    #ax.set_extent([5, 15, 47, 55], crs=ccrs.PlateCarree())
    # Exclude 'EU' from the network lines and links
    network.lines = network.lines[~(network.lines.bus0.str.contains("EU") | network.lines.bus1.str.contains("EU"))]
    network.links = network.links[~(network.links.bus0.str.contains("EU") | network.links.bus1.str.contains("EU"))]

    plot_path = os.path.join(output_dir, "Load_distribution.png")
    plt.savefig(plot_path)
load_distribution(n)
#fertig aber ggf. Farbe ändern und Legende hinzufügen

def installed_capacity_distribution(network):
    #alle= techs = n.generators["carrier"].unique()
    techs = ['offwind-float', 'onwind', 'solar', 'ror']  # Technologien
    pretty_names = {
        'offwind-float': 'Floating Offshore Wind',
        'onwind': 'Onshore Wind',
        'solar': 'Solar Energy',
        'ror': 'Run-of-River Hydro'
    }

    line_color = 'gray'  # Linienfarbe
    tech_colors = {
        'offwind-float': 'lightblue',
        'onwind': 'green',
        'solar': 'gold',
        'ror': 'cyan'
    }

    network.lines = network.lines[
        ~(network.lines.bus0.str.contains("EU") | network.lines.bus1.str.contains("EU"))]
    network.links = network.links[
        ~(network.links.bus0.str.contains("EU") | network.links.bus1.str.contains("EU"))]

    n_graphs = len(techs)
    n_cols = 2
    n_rows = (n_graphs + n_cols - 1) // n_cols  # Rundet auf

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, subplot_kw={"projection": ccrs.EqualEarth()}
    )

    size = 6
    fig.set_size_inches(size * n_cols, size * n_rows)

    # Für die Punktgrößen in der Legende
    legend_sizes = [10, 50, 100]  # Installierte Leistung in MW
    size_scaling = 2e-5

    # Hier speichern wir die Handles für die Legende
    scatter_handles = []

    for i, tech in enumerate(techs):
        i_row = i // n_cols
        i_col = i % n_cols

        ax = axes[i_row, i_col]
        gens = network.generators[network.generators.carrier == tech]
        gen_distribution = (
            gens.groupby("bus").sum()["p_nom"].reindex(network.buses.index, fill_value=0.0)
        )
        gen_distribution = gen_distribution[~gen_distribution.index.str.contains("EU")]
        network.lines = network.lines[
            ~(network.lines.bus0.str.contains("EU") | network.lines.bus1.str.contains("EU"))]
        network.links = network.links[
            ~(network.links.bus0.str.contains("EU") | network.links.bus1.str.contains("EU"))]

        # Plot mit angepassten Farben
        network.plot(
            ax=ax,
            bus_sizes=2e-5 * gen_distribution,
            bus_colors=tech_colors.get(tech, 'gray'),  # Farben für Busse
            line_colors=line_color,  # Farbe für Linien
        )
        ax.set_title(pretty_names.get(tech, tech))
        ax.set_extent([5, 15, 47, 55], crs=ccrs.PlateCarree())

        # Scatter für installierte Kapazität auf den Bussen
        for bus, capacity in gen_distribution.items():
            # Erstelle einen Punkt für jede Generator-Kapazität
            ax.scatter(
                network.buses.loc[bus].x,  # x-Position der Busse
                network.buses.loc[bus].y,  # y-Position der Busse
                s=size_scaling * capacity,  # Punktgröße basierend auf der Kapazität
                color=tech_colors.get(tech, 'gray'),
                alpha=0.6,
            )

    # Legende mit verschiedenen Punktgrößen basierend auf installierter Kapazität
    legend_handles = []
    for size in legend_sizes:
        # Marker für die Legende, basierend auf der Kapazität
        legend_handle = mlines.Line2D([], [], linestyle='none', marker='o', color='gray',
                                      markersize=size_scaling * size, label=f"{size} MW")
        legend_handles.append(legend_handle)

    # Legende hinzufügen
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_sizes),
        title="Installed Capacity",
        fontsize='medium'
    )

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "Installed_capacity_distribution.png")
    plt.savefig(plot_path)
    plt.close()
installed_capacity_distribution(n)
#legende zeigt größe der Punkte nicht an

def generation_ugly(n):
    generators_to_keep = n.generators[n.generators.bus.str.contains("DE")].index
    p_by_carrier = n.generators_t.p[generators_to_keep].groupby(n.generators.carrier, axis=1).sum()
    p_by_carrier.drop(
        (p_by_carrier.max()[p_by_carrier.max() < 1700.0]).index, axis=1, inplace=True
    )
    p_by_carrier.columns
    colors = {
        "nuclear": "r",
        "offwind-float": "cyan",
        "onwind": "blue",
        "ror": "green",
        "solar": "yellow",
        "solar rooftop": "brown",
        "solar-hsat": "orange",
    }
    # reorder
    cols = [
        "nuclear",
        "offwind-float",
        "onwind",
        "ror",
        "solar",
        "solar rooftop",
        "solar-hsat",
    ]
    p_by_carrier = p_by_carrier[cols]

    c = [colors[col] for col in p_by_carrier.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    (p_by_carrier / 1e3).plot(kind="area", ax=ax, linewidth=4, color=c, alpha=0.7)
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylabel("GW")
    ax.set_xlabel("")
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "generation_ugly.png")
    plt.savefig(plot_path)


def merit_order(n):
   # technologies = n.generators["carrier"].unique()

    # Wenn du nur die Technologien anzeigen möchtest, die auch tatsächlich in Betrieb sind
   # active_technologies = technologies[technologies != "None"]  # Entfernt "None", falls es existiert


    df = n.generators[n.generators.index.str.contains("DE")]
    df_storage = n.storage_units[n.storage_units.index.str.contains("DE")]
 #   df_stores = n.stores[n.stores.index.str.contains("DE")]
    combined_df = pd.concat([df, df_storage], axis=0, ignore_index=True)


    # Sort DataFrame by marginal_cost
    df = combined_df.sort_values(by=['marginal_cost'])
    df = df.reset_index(drop=True)
    df['p_nom'] = df['p_nom'] / 1000

    df["xpos"] = ''
    cumulative_xpos = 0  # Initialize cumulative xpos
    for i, row in df.iterrows():
        df.loc[i, "xpos"] = cumulative_xpos + row["p_nom"] / 2
        cumulative_xpos += row["p_nom"]

    df["xpos"] = pd.to_numeric(df["xpos"])

    color_dict = {
        'ror': '#005F73',
        'offwind-float': '#008FC8',
       # 'solar-hsat': '#52B695',
        'onwind': '#85EBFF',
        'solar': '#FDE88D',
        'nuclear': '#A1A1A1',
        #'oil primary': '#001219',
       # 'coal': '#FC7E46',
       # 'biogas': '#FC7E46',
       # 'gas': '#9B2226',
       # 'urban central solar thermal': '#9B2226',
        #'solar rooftop': '#0A9396',
        #'rural solar thermal': '#F6B3AC',
       # 'urban decentral solar thermal': '#EE6E60',
        'PHS': '#B0A9FF',
        'hydro': '#006E6D',
        #'H2 Store': '#D1D4E2',
        #'battery': '#F7A8B8',
        #'EV battery': '#E1D7D1',
        #'urban central water tanks': '#C1D8E4',
        #'home battery': '#F2D6A1',
        #'rural water tanks': '#A9F4F1',
        #'urban decentral water tanks': '#F3E3C0'
    }

    def add_color(xpos):
        carrier = df.loc[df['xpos'] == xpos, 'carrier'].values[0]
        color = color_dict.get(carrier)
        return color

    plt.figure(figsize=(15, 1))
    plt.rcParams["font.size"] = 12
   # plt.rcParams["font.family"] = "serif"
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1
    plt.rcParams['figure.figsize'] = [10, 4]

    xpos = df["xpos"].values.tolist()
    y = [val + 1 if val < 10 else val for val in df["marginal_cost"].values.tolist()]
    #y = df["marginal_cost"].values.tolist()
    w = df["p_nom"].values.tolist()

    colors = [add_color(x) for x in xpos]

    fig, ax = plt.subplots()

    for i, x in enumerate(xpos):
        ax.bar(x, y[i], width=w[i], color=colors[i])

    plt.xlim(0, df["p_nom"].sum())
    plt.ylim(0, 14)

    technology_names = {
        'ror': 'Run-of-River Hydro',
        'offwind-float': 'Floating Offshore Wind',
       # 'solar-hsat': 'High-Altitude Solar Thermal',
        'onwind': 'Onshore Wind',
        'solar': 'Solar Energy',
        'nuclear': 'Nuclear Power',
       # 'oil primary': 'Oil Primary Energy',
       # 'coal': 'Coal',
       # 'biogas': 'Biogas',
       # 'gas': 'Natural Gas',
       # 'urban central solar thermal': 'Urban Central Solar Thermal',
       # 'solar rooftop': 'Solar Rooftop Installations',
       # 'rural solar thermal': 'Rural Solar Thermal',
       # 'urban decentral solar thermal': 'Urban Decentral Solar Thermal',
        'PHS': 'Pumped Hydro Storage',
        'hydro': 'Hydropower',
        #'H2 Store': 'Hydrogen Storage',
        #'battery': 'Battery Storage',
        #'EV battery': 'Electric Vehicle Battery',
        #'urban central water tanks': 'Urban Central Water Tanks',
        #'home battery': 'Home Battery Storage',
        #'rural water tanks': 'Rural Water Tanks',
        #'urban decentral water tanks': 'Urban Decentral Water Tanks'
    }
    legend_elements = [mpatches.Patch(color=color_dict[carrier], label=technology_names[carrier]) for carrier in color_dict]
    plt.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False)
    plt.xlabel("Installed Capacity (GW)")

    plt.ylabel("Marginal cost (€/MWh)")
    plt.grid(axis='y')
    plt.box(False)
    plt.savefig("Merit_Order_Final.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    plot_path = os.path.join(output_dir, "merit_order.png")
    plt.savefig(plot_path)
    plt.close()
merit_order(n)
#fertig

def average_usage_lines(n):
    max_usage = n.lines[~(network.lines.bus0.str.contains("EU") | network.lines.bus1.str.contains("EU"))]
    max_usage["max_usage"] = max_usage.s_max_pu * max_usage.s_nom_opt
    n.lines = network.lines[~(network.lines.bus0.str.contains("EU") | network.lines.bus1.str.contains("EU"))]
    n.links = network.links[~(network.links.bus0.str.contains("EU") | network.links.bus1.str.contains("EU"))]



    average_usage = n.lines_t.p0.abs().mean()
    average_usage = pd.DataFrame(average_usage, columns=['average_usage'])
    merged_usage = pd.merge(max_usage, average_usage, left_index=True, right_index=True)
    merged_usage["percentage_usage_line"] = merged_usage.average_usage / merged_usage.max_usage
    series_usage = merged_usage["percentage_usage_line"]
   # series_usage.loc['160'] = 0
   # series_usage.loc['183'] = 0
    flow = pd.Series(1, index=n.branches().index) / 1000


    collection = n.plot(
        bus_sizes=0.02,
        bus_colors='#9B2226',  # gen / 5e3,
        # bus_colors={"gas": "indianred", "wind": "midnightblue"},
        margin=0.5,
        flow=flow,
        line_widths=1,
        link_widths=1,
        projection=ccrs.EqualEarth(),
        color_geomap=True,
        # line_colors=n.lines_t.p0.mean().abs(),
        line_colors=series_usage * 100
    )
    custom_colors = ['#000000', '#FF0000', '#FFFF00', '#00FF00']  # Add the colors you want
   # custom_cmap = colors.ListedColormap(custom_colors)
    plt.rcParams["font.size"] = 9
    #plt.rcParams["font.family"] = "serif"
    plt.colorbar(collection[2], fraction=0.04, pad=0.02, label="Average usage of line capacity [%]")#, cmap=custom_cmap)
    ax = plt.gca()
    ax.set_extent([5, 15, 47, 55], crs=ccrs.PlateCarree()) # Specify the extent here
    #plt.savefig("Line_usage_Redispatch.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig('Line_usage_2BZ_withOBZ.pdf')
    plt.show()
    plot_path = os.path.join(output_dir, "line_usage.png")
    plt.savefig(plot_path)
average_usage_lines(n)

def multiple_dispatch_results(n):
    #zum vergleichen der zwei szenarien?
    y=n

    results = n.generators_t.p[[col for col in n.generators_t.p.columns if 'DE' in col]].reset_index(drop=True) * 24
    merged_df = pd.merge(results.T, n.generators["bus"], on='Generator')
    # results = n.generators_t.p.reset_index(drop=True)*24
    df_deii1 = merged_df[merged_df['bus'] == 'DEII1'].T
    results = df_deii1.drop('bus')

    df = n.storage_units_t.p[[col for col in n.storage_units_t.p.columns if 'DE' in col]].reset_index(drop=True) * 24
    # df = n.storage_units_t.p.reset_index(drop=True)*24
    merged_df = pd.merge(df.T, n.storage_units["bus"], on='StorageUnit')
    # results = n.generators_t.p.reset_index(drop=True)*24
    storage_merged = merged_df[merged_df['bus'] == 'DEII1'].T
    storage_merged = storage_merged.drop('bus')
    storage_dispatch_pos = storage_merged[storage_merged > 0]
    storage_dispatch_neg = storage_merged[storage_merged < 0]

    df_store = n.stores_t.p[[col for col in n.stores_t.p.columns if 'DE' in col]].reset_index(drop=True) * 24
    # df_store = n.stores_t.p.reset_index(drop=True)*24
    merged_df = pd.merge(df_store.T, n.stores["bus"], on='Store')
    # results = n.generators_t.p.reset_index(drop=True)*24
    store_merged = merged_df[merged_df['bus'] == 'DEII1'].T
    store_merged = store_merged.drop('bus')
    store_pos = df_store[df_store > 0]
    store_neg = df_store[df_store < 0]
    # Zweiter
    results2 = y.generators_t.p[[col for col in y.generators_t.p.columns if 'DE' in col]].reset_index(drop=True) * 24
    # results2 = y.generators_t.p.reset_index(drop=True)*24
    merged_df = pd.merge(results2.T, n.generators["bus"], on='Generator')
    # results = n.generators_t.p.reset_index(drop=True)*24
    df_deii2 = merged_df[merged_df['bus'] == 'DEII2'].T
    results2 = df_deii2.drop('bus')

    df2 = y.storage_units_t.p[[col for col in y.storage_units_t.p.columns if 'DE' in col]].reset_index(drop=True) * 24
    # df2 = y.storage_units_t.p.reset_index(drop=True)*24
    merged_df = pd.merge(df2.T, n.storage_units["bus"], on='StorageUnit')
    storage_merged = merged_df[merged_df['bus'] == 'DEII2'].T
    storage_merged = storage_merged.drop('bus')
    storage_dispatch_pos2 = df2[df2 > 0]
    storage_dispatch_neg2 = df2[df2 < 0]

    df_store2 = y.stores_t.p[[col for col in y.stores_t.p.columns if 'DE' in col]].reset_index(drop=True) * 24
    # df_store2 = y.stores_t.p.reset_index(drop=True)*24
    merged_df = pd.merge(df_store2.T, n.stores["bus"], on='Store')
    # results = n.generators_t.p.reset_index(drop=True)*24
    store_merged = merged_df[merged_df['bus'] == 'DEII2'].T
    store_merged = store_merged.drop('bus')
    store_pos2 = df_store2[df_store2 > 0]
    store_neg2 = df_store2[df_store2 < 0]

    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("DEII1", "DEII2"))

    fig.add_trace(
        go.Scatter(x=results.index,
                   y=results[[col for col in results.columns if (("CCGT" in col) or ("OCGT" in col))]].sum(
                       axis=1) / 1e6, name='Gas', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#9B2226"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'oil' in col]].sum(axis=1) / 1e6,
                   name='Oil', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#001219"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'coal' in col]].sum(axis=1) / 1e6,
                   name='Coal', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#FC7E46"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'nuclear' in col]].sum(axis=1) / 1e6,
                   name='Nuclear', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#A1A1A1"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index,
                   y=results[[col for col in results.columns if 'geothermal' in col]].sum(axis=1) / 1e6,
                   name='Geothermal', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#E9D8A6"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'biomass' in col]].sum(axis=1) / 1e6,
                   name='Biomass', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#30725C"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'onwind' in col]].sum(axis=1) / 1e6,
                   name='Onshore wind', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor='#85EBFF'), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'offwind' in col]].sum(axis=1) / 1e6,
                   name='Offshore wind', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#008FC8"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'solar' in col]].sum(axis=1) / 1e6,
                   name='Solar', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#FDE88D"), row=1, col=1)
    # export from Storage
    fig.add_trace(
        go.Scatter(x=storage_dispatch_pos.index,
                   y=storage_dispatch_pos[[col for col in storage_dispatch_pos.columns if "hydro" in col]].sum(
                       axis=1) / 1e6, name='Hydro', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#52B695"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=storage_dispatch_pos.index,
                   y=storage_dispatch_pos[[col for col in storage_dispatch_pos.columns if 'PHS' in col]].sum(
                       axis=1) / 1e6, name='PHS', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#0A9396"), row=1, col=1)
    # speichern
    fig.add_trace(
        go.Scatter(x=storage_dispatch_neg.index,
                   y=storage_dispatch_neg[[col for col in storage_dispatch_neg.columns if "hydro" in col]].sum(
                       axis=1) / 1e6, name='Hydro', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#52B695"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=storage_dispatch_neg.index,
                   y=storage_dispatch_neg[[col for col in storage_dispatch_neg.columns if 'PHS' in col]].sum(
                       axis=1) / 1e6, name='PHS', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#0A9396"), row=1, col=1)

    # export store
    fig.add_trace(
        go.Scatter(x=store_pos.index, y=store_pos[[col for col in store_pos.columns if 'H2' in col]].sum(axis=1) / 1e6,
                   name='H2', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#F6B3AC"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=store_pos.index,
                   y=store_pos[[col for col in store_pos.columns if 'battery' in col]].sum(axis=1) / 1e6,
                   name='Battery', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#EE6E60"), row=1, col=1)
    # speichern
    fig.add_trace(
        go.Scatter(x=store_neg.index, y=store_neg[[col for col in store_neg.columns if 'H2' in col]].sum(axis=1) / 1e6,
                   name='H2', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#F6B3AC"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=store_neg.index,
                   y=store_neg[[col for col in store_neg.columns if 'battery' in col]].sum(axis=1) / 1e6,
                   name='Battery', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#EE6E60"), row=1, col=1)

    # Plot 2
    fig.add_trace(
        go.Scatter(x=results2.index,
                   y=results2[[col for col in results2.columns if (("CCGT" in col) or ("OCGT" in col))]].sum(
                       axis=1) / 1e6, name='Gas', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#9B2226"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index, y=results2[[col for col in results2.columns if 'oil' in col]].sum(axis=1) / 1e6,
                   name='Oil', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#001219"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index, y=results2[[col for col in results2.columns if 'coal' in col]].sum(axis=1) / 1e6,
                   name='Coal', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#FC7E46"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index,
                   y=results2[[col for col in results2.columns if 'nuclear' in col]].sum(axis=1) / 1e6, name='Nuclear',
                   fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#A1A1A1"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index,
                   y=results2[[col for col in results2.columns if 'geothermal' in col]].sum(axis=1) / 1e6,
                   name='Geothermal', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#E9D8A6"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index,
                   y=results2[[col for col in results2.columns if 'biomass' in col]].sum(axis=1) / 1e6, name='Biomass',
                   fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#30725C"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index, y=results2[[col for col in results2.columns if 'onwind' in col]].sum(axis=1) / 1e6,
                   name='Onshore wind', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor='#85EBFF'), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index,
                   y=results2[[col for col in results2.columns if 'offwind' in col]].sum(axis=1) / 1e6,
                   name='Offshore wind', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#008FC8"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=results2.index, y=results2[[col for col in results2.columns if 'solar' in col]].sum(axis=1) / 1e6,
                   name='Solar', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#FDE88D"), row=1, col=2)

    # export from Storage
    fig.add_trace(
        go.Scatter(x=storage_dispatch_pos2.index,
                   y=storage_dispatch_pos2[[col for col in storage_dispatch_pos2.columns if "hydro" in col]].sum(
                       axis=1) / 1e6, name='Hydro', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#52B695"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=storage_dispatch_pos2.index,
                   y=storage_dispatch_pos2[[col for col in storage_dispatch_pos2.columns if 'PHS' in col]].sum(
                       axis=1) / 1e6, name='PHS', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#0A9396"), row=1, col=2)
    # speichern
    fig.add_trace(
        go.Scatter(x=storage_dispatch_neg2.index,
                   y=storage_dispatch_neg2[[col for col in storage_dispatch_neg2.columns if "hydro" in col]].sum(
                       axis=1) / 1e6, name='Hydro', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#52B695"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=storage_dispatch_neg2.index,
                   y=storage_dispatch_neg2[[col for col in storage_dispatch_neg2.columns if 'PHS' in col]].sum(
                       axis=1) / 1e6, name='PHS', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#0A9396"), row=1, col=2)

    # export store
    fig.add_trace(
        go.Scatter(x=store_pos2.index,
                   y=store_pos2[[col for col in store_pos2.columns if 'H2' in col]].sum(axis=1) / 1e6, name='H2',
                   fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#F6B3AC"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=store_pos2.index,
                   y=store_pos2[[col for col in store_pos2.columns if 'battery' in col]].sum(axis=1) / 1e6,
                   name='Battery', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#EE6E60"), row=1, col=2)
    # speichern
    fig.add_trace(
        go.Scatter(x=store_neg2.index,
                   y=store_neg2[[col for col in store_neg2.columns if 'H2' in col]].sum(axis=1) / 1e6, name='H2',
                   fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#F6B3AC"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=store_neg2.index,
                   y=store_neg2[[col for col in store_neg2.columns if 'battery' in col]].sum(axis=1) / 1e6,
                   name='Battery', fill='tozeroy', stackgroup='two', mode='none',
                   fillcolor="#EE6E60"), row=1, col=2)

    legend_items = [
        ('Gas', '#9B2226'),
        ('Biomass', '#30725C'),
        ('Coal', '#FC7E46'),
        ('RoR', '#005F73'),
        # ('Nuclear', '#A1A1A1'),
        ('Onwind', '#85EBFF'),
        ('Solar', '#FDE88D'),
        ('Oil', '#001219'),
        ('Geothermal', '#E9D8A6'),
        ('Hydro', '#52B695'),
        ('PHS', '#0A9396'),
        ('Battery', '#EE6E60'),
        ('H2', '#F6B3AC'),
        ('Offwind', '#008FC8'),
        ('Offwind', '#008FC8')
    ]

    # Update xaxis properties
    fig.update_xaxes(range=[0, 365], row=1, col=1)
    fig.update_xaxes(range=[0, 365], row=1, col=2)
    # Update yaxis properties
    fig.update_yaxes(range=[-0.75, 1.3], row=1, col=1)
    fig.update_yaxes(range=[-0.75, 1.3], row=1, col=2)

    fig.update_xaxes(title_text="Timesteps", title_font=dict(size=16))
    fig.update_yaxes(gridcolor="rgba(166, 166, 166, 0.5)")
    fig.update_layout(
        showlegend=False,
        yaxis=dict(title='Generation [TWh]', title_font=dict(size=16)),
        font=dict(size=16, family="Times New Roman"),
        margin=dict(l=10, r=10, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,  # adjust the height of the figure
        width=800,  # adjust the width of the figure
        font_color='black')
    fig.write_image('dispatch_BZ2_noOBZ_DEII1_DEII2.svg')
    fig.show()

def dispatch_single(n):

    # storage_dispatch = pd.read_csv('/home/marie/pypsa-eur2.0/pypsa-eur_marie/Z-Finale_Ergebnsise/mit_OBZ/4_BZ/market_model/storage_units-p_dispatch.csv')
    # storage_store = pd.read_csv('/home/marie/pypsa-eur2.0/pypsa-eur_marie/Z-Finale_Ergebnsise/mit_OBZ/4_BZ/market_model/storage_units-p_store.csv')
    #stores_level = pd.read_csv('/home/marie/pypsa-eur2.0/pypsa-eur_marie/Z-Finale_Ergebnsise/mit_OBZ/4_BZ/market_model/stores-p.csv')
    #demand = pd.read_csv('/home/marie/pypsa-eur2.0/pypsa-eur_marie/Z-Finale_Ergebnsise/mit_OBZ/4_BZ/market_model/loads-p.csv')

    n.lines = network.lines[~(network.lines.bus0.str.contains("EU") | network.lines.bus1.str.contains("EU"))]
    n.links = network.links[~(network.links.bus0.str.contains("EU") | network.links.bus1.str.contains("EU"))]

    results = n.generators_t.p.loc[:, ~n.generators_t.p.columns.str.contains("EU")]

    #storage_units_p = n.storage_units[~n.generators.index.str.contains("EU")]
    # storage_dispatch = pd.read_csv('/home/marie/pypsa-eur2.0/pypsa-eur_marie/Z-Finale_Ergebnsise/mit_OBZ/4_BZ/market_model/storage_units-p_dispatch.csv')
    # storage_store = pd.read_csv('/home/marie/pypsa-eur2.0/pypsa-eur_marie/Z-Finale_Ergebnsise/mit_OBZ/4_BZ/market_model/storage_units-p_store.csv')
    #stores_level = stores-p.csv')
    #demand = n.loads_t
    fig = go.Figure()
  #  fig.add_trace(
  #      go.Scatter(x=results.index,
  #                 y=results[[col for col in results.columns if ("gas" in col) ]].sum(
  #                     axis=1) / 1000, name='Gas', fill='tonexty', stackgroup='one', mode='none',
  #                 fillcolor="#9B2226"))
#    fig.add_trace(
#        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'oil' in col]].sum(axis=1) / 1000,
#                   name='Oil', fill='tonexty', stackgroup='one', mode='none',
#                   fillcolor="#001219"))
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'ror' in col]].sum(axis=1) / 1000,
                   name='Coal', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#FC7E46"))
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'nuclear' in col]].sum(axis=1) / 1000,
                   name='Nuclear', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#A1A1A1"))
#    fig.add_trace(
#        go.Scatter(x=results.index,
#                   y=results[[col for col in results.columns if 'geothermal' in col]].sum(axis=1) / 1000,
#                   name='Geothermal', fill='tonexty', stackgroup='one', mode='none',
#                   fillcolor="#E9D8A6"))
#    fig.add_trace(
#        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'biomass' in col]].sum(axis=1) / 1000,
#                   name='Biomass', fill='tonexty', stackgroup='one', mode='none',
#                   fillcolor="#30725C"))
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'onwind' in col]].sum(axis=1) / 1000,
                   name='Onshore wind', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor='#85EBFF'))
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'offwind' in col]].sum(axis=1) / 1000,
                   name='Offshore wind', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#008FC8"))
    fig.add_trace(
        go.Scatter(x=results.index, y=results[[col for col in results.columns if 'solar' in col]].sum(axis=1) / 1000,
                   name='Solar', fill='tonexty', stackgroup='one', mode='none',
                   fillcolor="#FAFE58"))
  #  # export from Storage
  #  fig.add_trace(
  #      go.Scatter(x=storage_units_p.index,
  #                 y=storage_units_p[[col for col in storage_units_p.columns if "hydro" in col]].sum(axis=1) / 1000,
  #                 name='Hydro', fill='tonexty', stackgroup='one', mode='none',
  #                 fillcolor="#52B695"))
  #  fig.add_trace(
  #      go.Scatter(x=storage_units_p.index,
  #                 y=storage_units_p[[col for col in storage_units_p.columns if 'PHS' in col]].sum(axis=1) / 1000,
  #                 name='PHS', fill='tonexty', stackgroup='one', mode='none',
  #                 fillcolor="#0A9396"))
    # #storing
    # fig.add_trace(
    #     go.Scatter(x=storage_store.index, y=-storage_store[[col for col in storage_store.columns if 'PHS' in col ]].sum(axis=1)/1000, name='PHS store', fill='tozeroy', stackgroup='two', mode='none',
    #             fillcolor="#0A9396"))
    # #battery
    # fig.add_trace(
    #     go.Scatter(x=storage_store.index, y=-storage_store[[col for col in storage_store.columns if 'battery' in col ]].sum(axis=1)/1000, name='Battery store', fill='tozeroy', stackgroup='two', mode='none',
    #             fillcolor="#EE6E60"))
    # demand
    # fig.add_trace(
    # go.Scatter(x=demand.index, y=demand.sum(axis=1)/1000, name='Demand', stackgroup='two', mode='lines',
    #         line=dict(color="#AB3F9E")))

    # #legend_items = [
    # #    ('Gas', '#9B2226'),
    # #    ('Biomass', '#30725C'),
    # #    ('Coal', '#FC7E46'),
    # #    ('RoR', '#005F73'),
    #     #('Nuclear', '#A1A1A1'),
    # #    ('Onwind', '#85EBFF'),
    #     ('Solar', '#FAFE58'),
    #     ('Oil', '#001219')
    #     ('Geothermal','#E9D8A6'),
    #     ('Hydro', '#52B695'),
    #     ('PHS', '#0A9396'),
    #     ('Battery', '#EE6E60'),
    #     ('H2', '#F6B3AC'),
    #     ('Offwind', '#008FC8')
    # ]
    # Set the number of items per row and spacing between the items
    # fig.update_layout(showlegend=False)
    # items_per_row = 7
    # spacing = 0.09
    # num_rows = (len(legend_items) + items_per_row - 1) // items_per_row  # Calculate the number of rows

    # for i, (name, color) in enumerate(legend_items):
    #     row_index = i // items_per_row  # Calculate the row index
    #     col_index = i % items_per_row  # Calculate the column index

    #     x_position = col_index * spacing + col_index * 0.1
    #     if row_index >= 2:
    #         x_position = col_index * spacing + (spacing * items_per_row)
    #     y_position = -row_index * 0.1 - 0.4

    #     fig.add_annotation(
    #         x=x_position,
    #         y=y_position,
    #         #xref='paper',
    #         #yref='paper',
    #         text=f"<span style='color: {color};'>■</span> {name}",
    #         showarrow=False,
    #         font=dict(size=16),
    #         bgcolor='rgba(0,0,0,0)',
    #         bordercolor='rgba(0,0,0,0)',
    #     )

    fig.update_layout(
        legend_tracegroupgap=5,
        #xaxis=dict(title='Timesteps'),#, dtick=24, title_font=dict(size=16)),
        xaxis=dict(
            title='Timesteps',
            tickvals=[results.index[0], results.index[-1]],  # Zeitpunkte
            ticktext=['Januar 2040','Dezember 2040'],#Januar 2040', 'April 2040', 'Dezember 2040'],  # Angezeigte Werte
            title_font=dict(size=16)
        ),
        yaxis=dict(title='Generation [GWh]'),#, title_font=dict(size=16), range=[-50, 220]),
        #font=dict(size=16, family="Times New Roman"),
        legend=dict(x=0, y=-0.15, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),#, font_size=22),
        legend_orientation="h",
        margin=dict(l=10, r=10, t=15, b=200),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor="rgba(166, 166, 166, 0.5)",
        height=500,  # adjust the height of the figure
        width=700,  # adjust the width of the figure
        font_color='black')


    fig.show()
    fig.write_image(os.path.join(output_dir,'dispatch.png'))

def map_generation_pie_charts(n):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())})
    n.generators = n.generators[n.generators.index.str.contains("DE")]
    n.storage_units = n.storage_units[n.storage_units.index.str.contains("DE")]
    gen = n.generators[n.generators.carrier != "load"].groupby(["bus", "carrier"]).p_nom_opt.sum()
    sto = n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum()
    buses = pd.concat([gen, sto])

    #bus_scale = 6e3
    bus_scale = 1e5
    line_scale = 6e3
    #bus_sizes = [100, 1000]  # in MW
    bus_sizes = [1, 10]
    line_sizes = [100, 1000]  # in MW


    with plt.rc_context({"patch.linewidth": 0.}):
        n.plot(
            bus_sizes=buses / bus_scale,
            bus_alpha=0.7,
            line_widths=n.lines.s_nom_opt / line_scale,
            link_widths=n.links.p_nom_opt / line_scale,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )
    #regions_onshore.plot(
    #    ax=ax,
    #    facecolor="whitesmoke",
    #    edgecolor="white",
    #    aspect="equal",
     #   transform=ccrs.PlateCarree(),
    #    linewidth=0,
    #)
    #ax.set_extent(regions_onshore.total_bounds[[0, 2, 1, 3]])
    #legend_kwargs = {"loc": "upper left", "frameon": False}
    #legend_circles_dict = {"bbox_to_anchor": (1, 0.67), "labelspacing": 2.5, **legend_kwargs}

    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1000} GW" for s in bus_sizes],
        legend_kw=legend_circles_dict,
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1000} GW" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (1, 0.8), **legend_kwargs},
    )

    n.carriers.color = n.carriers.color.dropna()  # Entfernt NaN-Werte
    n.carriers.color = n.carriers.color[n.carriers.color != '']

  #  add_legend_patches(
  #      ax,
  #      n.carriers.color.fillna('#000000')  ,
  #      n.carriers.nice_name,
  #      legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"},
  #  )
    #fig.tight_layout()

    plt.show()
    plot_path = os.path.join(output_dir, "map_generation_pie_chart.png")
    plt.savefig(plot_path)


##################Alter kram#############################
def alt_bearbeiten():

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 8))

    plt.hexbin(
        n.buses.x,
        n.buses.y,
        gridsize=20,
        C=n.buses_t.marginal_price.loc[now],
        cmap=plt.cm.jet,
        zorder=3,
    )
    n.plot(ax=ax, line_widths=pd.Series(0.5, n.lines.index), bus_sizes=0)

    cb = plt.colorbar(location="bottom")
    cb.set_label("Locational Marginal Price (EUR/MWh)")
    fig.tight_layout()





    n = pypsa.Network("results/postnetworks/base_s_10_lvopt___2040.nc")

    n.plot()  # plots network
    plot_path = os.path.join(output_dir, "plot1.png")
    plt.savefig(plot_path)

    # shows all the components
    for c in n.iterate_components(list(n.components.keys())[2:]):
        print("Component '{}' has {} entries".format(c.name, len(c.df)))

    n.snapshots
    len(n.snapshots)
    # input data
    n.lines.head()
    n.generators.head()
    n.storage_units.head()
    n.loads.head()  # mostly time dependent data, therefore mostly empty
    n.loads_t.p_set.head()  # time series power demand in MW for each node
    n.loads_t.p_set.head(axis=1).plot(figsize=(15, 3))  # load
    plot_path = os.path.join(output_dir, "plot2.png")
    plt.savefig(plot_path)


    n.generators_t.p_max_pu.head()
    n.generators_t.p_max_pu.loc["2013-07", "DE0 0 solar"].plot(15, 3)
    plot_path = os.path.join(output_dir, "plot3.png")
    plt.savefig(plot_path)

    n.generators_t.p_max_pu.loc["2013-07":"2013-09", "DE0 0 onwind"].plot(figsize=(15, 3));
    plot_path = os.path.join(output_dir, "plot4.png")
    plt.savefig(plot_path)

    n.objective / 1e9  # billion euros p.a. // Total annual system costs

    (n.lines.s_nom_opt - n.lines.s_nom).head()  # Transmission line expansion

    n.generators.groupby("carrier").p_nom_opt.sum() / 1e3  # GW Optimal Generator/Storage Capacity
    n.storage_units.groupby("carrier").p_nom_opt.sum() / 1e3  # GW

    # Energy Storage
    (n.storage_units_t.state_of_charge.sum(axis=1).resample("D").mean() / 1e6).plot(figsize=(15, 3))
    plot_path = os.path.join(output_dir, "plot5.png")
    plt.savefig(plot_path)
    # how do batteries behave in the system
    (n.storage_units_t.state_of_charge["2013-07"].filter(like="battery", axis=1).sum(axis=1) / 1e6).plot(figsize=(15, 3))
    plot_path = os.path.join(output_dir, "plot6.png")
    plt.savefig(plot_path)
    # H2
    (n.storage_units_t.state_of_charge.filter(like="H2", axis=1).sum(axis=1) / 1e6).plot(figsize=(15, 3))
    plot_path = os.path.join(output_dir, "plot7.png")
    plt.savefig(plot_path)
    # hydrogen
    (n.storage_units_t.state_of_charge.filter(like="hydro", axis=1).sum(axis=1) / 1e6).plot(figsize=(15, 3))
    plot_path = os.path.join(output_dir, "plot8.png")
    plt.savefig(plot_path)





    loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index()).fillna(0.)

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw = {"projection": ccrs.PlateCarree()}  # ccrs.Mercator() = Karte von oben
        )

    n.plot(ax=ax,
           bus_colors="gray",
           branch_components=["Line"],
           line_widths=n.lines.s_nom_opt / 3e3,
           line_colors=loading,
           line_cmap=plt.cm.viridis,
           color_geomap=True,
           bus_sizes=0)
    ax.axis("off")
    plot_path = os.path.join(output_dir, "plot9.png")
    plt.savefig(plot_path)



    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs  # Für Kartenprojektionen
    import pypsa  # Für das Netzwerkobjekt
    import pandas as pd  # Für den Umgang mit tabellarischen Daten

    # Annahme: 'network' ist bereits definiert und geladen
    # Annahme: 'now' ist ein gültiger Zeitstempel in `network.buses_t.marginal_price`
    network = n
    # Erstellung der Figur und Achse mit einer Kartenprojektion
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 8))

    # Hexbin-Plot der marginalen Preise an den Bus-Standorten
    plt.hexbin(
        network.buses.x,  # Längengrade der Busse
        network.buses.y,  # Breitengrade der Busse
        gridsize=20,  # Größe des Hexbin-Rasters
        C=network.buses_t.marginal_price.loc[now],  # Marginale Preise für den aktuellen Zeitpunkt
        cmap=plt.cm.jet,  # Farbkarte
        zorder=3,  # Zeichnungsreihenfolge
    )

    # Netzwerk auf der Karte zeichnen
    network.plot(ax=ax, line_widths=pd.Series(0.5, network.lines.index), bus_sizes=0)

    # Hinzufügen einer Farbskala (Colorbar)
    cb = plt.colorbar(ax.collections[0], ax=ax, location="bottom")  # Farbskala für den Hexbin-Plot
    cb.set_label("Locational Marginal Price (EUR/MWh)")

    # Layout der Figur anpassen
    fig.tight_layout()

    # Plot anzeigen oder speichern
    plt.show()
    # fig.savefig("output_plot.png")  # Optional: Plot speichern
