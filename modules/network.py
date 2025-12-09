from modules import *

class Network:
    def __init__(self, sn, en, t0, C, on, dn, q_od, node_coords=None, name=None):
        self.sn = sn
        self.en = en
        self.t0 = t0
        self.C = C
        self.on = on
        self.dn = dn
        self.q_od = q_od
        self.node_coords = node_coords
        self.name = name

    def summary(self):
        print(f"Network: {len(self.sn)} links, {len(set(self.sn) | set(self.en))} nodes, {len(self.on)} OD pairs")

    def plot(self, with_labels=True, node_size=300):
        G = nx.DiGraph()
        for i in range(len(self.sn)):
            G.add_edge(self.sn[i], self.en[i], weight=self.t0[i])
        if self.node_coords:
            pos = {node: (x, y) for node, (x, y) in self.node_coords.items()}
        else:
            pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=with_labels, node_size=node_size, arrows=True)
        plt.title("Network visualization")
        plt.show()
   
def load_network(name):
    if name == "toy":
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "data", "toy_model")

        df_net = pd.read_csv(os.path.join(data_dir, "toy_net.csv"))
        df_od = pd.read_csv(os.path.join(data_dir, "toy_od.csv"))
        
        sn = df_net['init_node'].to_numpy(dtype=int)
        en = df_net['term_node'].to_numpy(dtype=int)
        t0 = df_net['free_flow_time'].to_numpy()
        C = df_net['capacity'].to_numpy()

        on = df_od['O'].to_numpy(dtype=int)
        dn = df_od['D'].to_numpy(dtype=int)
        q_od = df_od['Ton'].to_numpy()

        # Optionnel
        node_file = os.path.join(data_dir, "toy_node.csv")
        if os.path.exists(node_file):
            df_node = pd.read_csv(node_file)
            node_coords = dict(zip(df_node['Node'], zip(df_node['X'], df_node['Y'])))
        else:
            node_coords = None

        return Network(sn, en, t0, C, on, dn, q_od, node_coords=node_coords, name="toy")

    elif name == "sioux_falls":
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "data", "sioux_falls")
        
        df_net = pd.read_csv(os.path.join(data_dir, "SiouxFalls_net.csv"), sep=',')
        df_od = pd.read_csv(os.path.join(data_dir, "SiouxFalls_od.csv"), sep=',')
        df_node = pd.read_csv(os.path.join(data_dir, "SiouxFalls_node.csv"), sep=',')

        # Charger les liens depuis le .tntp
        tntp_file_path = os.path.join(data_dir, "SiouxFalls_net.tntp")
        df_links = load_tntp_links(tntp_file_path)

        sn = df_links['init_node'].to_numpy(dtype=int)
        en = df_links['term_node'].to_numpy(dtype=int)
        t0 = df_links['free_flow_time'].to_numpy()
        C = df_links['capacity'].to_numpy()

        on = df_od['O'].to_numpy(dtype=int)
        dn = df_od['D'].to_numpy(dtype=int)
        q_od = df_od['Ton'].to_numpy()

        node_coords = dict(zip(df_node['Node'], zip(df_node['X'], df_node['Y'])))

        # Charger le fichier des flux observés à l'équilibre
        flow_file_path = os.path.join(data_dir, "SiouxFalls_flow.tntp")
        if os.path.exists(flow_file_path):
            df_flow = load_tntp_flows(flow_file_path)
        else:
            df_flow = None

        net = Network(sn, en, t0, C, on, dn, q_od, node_coords=node_coords, name="sioux_falls")
        net.flow_ref = df_flow  # pour comparaison ultérieure
        return net
    
    elif name == "sioux_falls_reduced":
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "data", "sioux_falls_reduced")
        
        df_net = pd.read_csv(os.path.join(data_dir, "SiouxFalls_net_reduced.csv"), sep=',')
        df_od = pd.read_csv(os.path.join(data_dir, "SiouxFalls_od_reduced.csv"), sep=',')
        df_node = pd.read_csv(os.path.join(data_dir, "SiouxFalls_node_reduced.csv"), sep=',')

        # Charger les liens depuis le .tntp
        tntp_file_path = os.path.join(data_dir, "SiouxFalls_net_reduced.tntp")
        df_links = load_tntp_links(tntp_file_path)

        sn = df_links['init_node'].to_numpy(dtype=int)
        en = df_links['term_node'].to_numpy(dtype=int)
        t0 = df_links['free_flow_time'].to_numpy()
        C = df_links['capacity'].to_numpy()

        on = df_od['O'].to_numpy(dtype=int)
        dn = df_od['D'].to_numpy(dtype=int)
        q_od = df_od['Ton'].to_numpy()

        node_coords = dict(zip(df_node['Node'], zip(df_node['X'], df_node['Y'])))

        net = Network(sn, en, t0, C, on, dn, q_od, node_coords=node_coords)
        return net
    
    elif name == "barcelona":
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "data", "barcelona")
        
        # Charger les liens depuis le fichier .tntp
        tntp_net_path = os.path.join(data_dir, "Barcelona_net.tntp")
        df_links = load_tntp_links_barcelona(tntp_net_path)
        
        sn = df_links['init_node'].to_numpy(dtype=int)
        en = df_links['term_node'].to_numpy(dtype=int)
        t0 = df_links['free_flow_time'].to_numpy()
        C = df_links['capacity'].to_numpy()
        
        # Charger les données OD depuis le fichier .tntp
        tntp_trips_path = os.path.join(data_dir, "Barcelona_trips.tntp")
        df_od = load_tntp_trips_barcelona(tntp_trips_path)
        
        on = df_od['O'].to_numpy(dtype=int)
        dn = df_od['D'].to_numpy(dtype=int)
        q_od = df_od['demand'].to_numpy()
        
        # Charger les coordonnées des nœuds si disponibles
        node_file = os.path.join(data_dir, "Barcelona_node.csv")
        if os.path.exists(node_file):
            df_node = pd.read_csv(node_file)
            node_coords = dict(zip(df_node['Node'], zip(df_node['X'], df_node['Y'])))
        else:
            node_coords = None
        
        net = Network(sn, en, t0, C, on, dn, q_od, node_coords=node_coords, name="barcelona")
        return net


def load_tntp_links(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("~") and "init_node" in line:
            data_start_index = i + 1
            break
    data_lines = lines[data_start_index:]
    records = []
    for line in data_lines:
        if line.strip() == "" or line.strip().startswith("~"):
            continue
        line_clean = line.strip().strip(";").strip()
        if line_clean:
            parts = line_clean.split()
            if len(parts) == 10:
                records.append([int(parts[0]), int(parts[1])] + [float(x) for x in parts[2:]])
    columns = [
        "init_node", "term_node", "capacity", "length", "free_flow_time",
        "b", "power", "speed", "toll", "link_type"
    ]
    return pd.DataFrame(records, columns=columns)

def load_tntp_flows(filepath):
    """
    Charge le fichier TNTP des flux d'équilibre (sioux_falls_flow.tntp)
    et renvoie un DataFrame contenant les colonnes :
        ['init_node', 'term_node', 'volume', 'cost']
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("~") and "From" in line:
            data_start_index = i + 1
            break

    if data_start_index is None:
        # Si le fichier n'a pas d'entête TNTP classique, on cherche "From"
        for i, line in enumerate(lines):
            if line.strip().startswith("From"):
                data_start_index = i + 1
                break

    if data_start_index is None:
        raise ValueError("Impossible de localiser le début des données dans le fichier TNTP.")

    data_lines = lines[data_start_index:]
    records = []
    for line in data_lines:
        if line.strip() == "" or line.strip().startswith("~"):
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                init_node = int(parts[0])
                term_node = int(parts[1])
                volume = float(parts[2])
                cost = float(parts[3])
                records.append([init_node, term_node, volume, cost])
            except ValueError:
                continue  # Ignore les lignes mal formatées

    df = pd.DataFrame(records, columns=["init_node", "term_node", "volume", "cost"])
    return df

def load_tntp_links_barcelona(filepath):
    """
    Charge le fichier Barcelona_net.tntp et retourne un DataFrame
    avec les colonnes: init_node, term_node, capacity, length, free_flow_time, etc.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Trouver le début des données (après <END OF METADATA>)
    data_start_index = None
    for i, line in enumerate(lines):
        if "<END OF METADATA>" in line:
            # Les données commencent après la ligne d'en-tête qui suit
            for j in range(i+1, len(lines)):
                if lines[j].strip().startswith("~") and "init_node" in lines[j]:
                    data_start_index = j + 1
                    break
            break
    
    if data_start_index is None:
        raise ValueError("Impossible de trouver le début des données dans le fichier")
    
    data_lines = lines[data_start_index:]
    records = []
    
    for line in data_lines:
        line = line.strip()
        # Ignorer les lignes vides ou les commentaires
        if not line or line.startswith("~") or line.startswith("<"):
            continue
        
        # Enlever le point-virgule final
        line = line.rstrip(";").strip()
        
        # Séparer les valeurs par des espaces/tabulations
        parts = line.split()
        
        if len(parts) >= 10:
            try:
                init_node = int(parts[0])
                term_node = int(parts[1])
                capacity = float(parts[2])
                length = float(parts[3])
                free_flow_time = float(parts[4])
                b = float(parts[5])
                power = float(parts[6])
                speed = float(parts[7])
                toll = float(parts[8])
                link_type = int(parts[9])
                
                records.append([init_node, term_node, capacity, length, free_flow_time,
                              b, power, speed, toll, link_type])
            except ValueError:
                continue  # Ignorer les lignes mal formatées
    
    columns = [
        "init_node", "term_node", "capacity", "length", "free_flow_time",
        "b", "power", "speed", "toll", "link_type"
    ]
    
    return pd.DataFrame(records, columns=columns)

def load_tntp_trips_barcelona(filepath):
    """
    Charge le fichier Barcelona_trips.tntp et retourne un DataFrame
    avec les colonnes: O (origine), D (destination), demand
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Trouver le début des données (après <END OF METADATA>)
    data_start_index = None
    for i, line in enumerate(lines):
        if "<END OF METADATA>" in line:
            data_start_index = i + 1
            break
    
    if data_start_index is None:
        raise ValueError("Impossible de trouver le début des données dans le fichier trips")
    
    data_lines = lines[data_start_index:]
    records = []
    current_origin = None
    
    for line in data_lines:
        line = line.strip()
        
        # Ignorer les lignes vides
        if not line:
            continue
        
        # Détecter une nouvelle origine
        if line.startswith("Origin"):
            parts = line.split()
            if len(parts) >= 2:
                current_origin = int(parts[1])
            continue
        
        # Parser les paires destination:demande
        if current_origin is not None:
            # Enlever le point-virgule final et séparer par point-virgule
            line = line.rstrip(";").strip()
            pairs = line.split(";")
            
            for pair in pairs:
                pair = pair.strip()
                if ":" in pair:
                    try:
                        dest_str, demand_str = pair.split(":")
                        destination = int(dest_str.strip())
                        demand = float(demand_str.strip())
                        records.append([current_origin, destination, demand])
                    except ValueError:
                        continue  # Ignorer les paires mal formatées
    
    df = pd.DataFrame(records, columns=["O", "D", "demand"])
    return df
