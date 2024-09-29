from pymongo import MongoClient
import pandas as pd
import dns.resolver
import certifi
from bson import ObjectId, errors
import config
import os
from datetime import datetime
import numpy as np

# DNS resolver and MongoDB connection setup
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']

client = MongoClient(f'mongodb+srv://{config.mongo_pat}', tlsCAFile=certifi.where())

# MongoDB collections
db = client['legacy-api-management']
col_items = db["items"]
col_bills = db["bills"]
col_users = db["users"]
col_soc = db["societies"]
col_billings = db["billings"]

airports_df = pd.read_csv("const/Air Airports and cities codes - O&D  2024.csv", delimiter=";")
rail_metro_df = pd.read_csv("const/O&D Metropolis rail 2024 v In&Out.csv",delimiter=";")
rail_euro_df = pd.read_csv("const/O&D European Rail 2024 v In&Out.csv",delimiter=";")
carrier_list = ["AF", "KL", "DL", "LU", "TO", "HV", "VS"]

def get_bills(id_soc,start_date):
    import pandas as pd
    from pymongo import MongoClient

    # Définir les critères de filtrage
    filter_criteria = {
        "societyId": f"{id_soc}",
        "createdAt": {"$gte": start_date},
        "type": {"$in": ["receipt", "credit", "unitary"]}
    }

    # Requête MongoDB pour extraire les informations des lignes
    projection = {
        "type": 1,  # Projeter 'type' à la racine
        "lines.itemId": 1,
        "lines.label": 1,
        "lines.type": 1,
        "lines.price.amount": 1,
        "lines.userIds": 1,
        "_id": 0  # Exclure le champ _id des résultats
    }

    # Effectuer la requête
    documents = col_bills.find(filter_criteria, projection)

    # Liste pour stocker les résultats
    data = []

    # Extraire les résultats et aplatir les valeurs des lignes
    for doc in documents:
        doc_type = doc.get("type")  # Récupérer le 'type' à la racine
        for line in doc.get('lines', []):
            # Extraire les valeurs des sous-documents
            item_data = {
                "ITEM_ID": line.get("itemId"),
                "type_bill": doc_type,  # Associer le type du document à chaque ligne
                "label": line.get("label"),
                "type": line.get("type"),
                "TOTAL_BILLED": line.get("price", {}).get("amount"),
                "userIds": ', '.join([str(user_id) for user_id in line.get("userIds", [])])
            }
            data.append(item_data)

    # Convertir la liste en DataFrame
    df = pd.DataFrame(data)

    df.to_csv(f"csv/base/bills_{id_soc}.csv")
    # Afficher le DataFrame
    print(df)


################ 🛫FLIGHT 🛫 #####################

def fetch_flight_details(item_id):
    # Chercher l'item dans la collection MongoDB
    item = col_items.find_one({"id": item_id})
    print(item_id)
    if item and item.get('type') == 'flight':
        trips = item.get('detail', {}).get('trips', [])
        all_legs_details = []

        # Boucle sur tous les trips
        for trip_index, trip in enumerate(trips):
            legs = trip.get('legs', [])

            if legs:
                # Si 'legs' existe, traiter chaque leg
                for leg_index, leg in enumerate(legs):
                    leg_details = {
                        f"{leg_index}_ori_city": leg.get('departure', {}).get('city', ''),
                        f"{leg_index}_des_city": leg.get('arrival', {}).get('city', ''),
                        f"{leg_index}_ori_country": leg.get('departure', {}).get('country', ''),
                        f"{leg_index}_des_country": leg.get('arrival', {}).get('country', ''),
                        f"{leg_index}_governing_carrier": leg.get('governingCarrier', '')
                    }
                    all_legs_details.append(leg_details)
            else:
                # Si 'legs' n'existe pas, utiliser les chemins alternatifs
                leg_details = {
                    f"{trip_index}_ori_city": trip.get('Dep', {}).get('IATALocationCode', ''),
                    f"{trip_index}_des_city": trip.get('Arrival', {}).get('IATALocationCode', ''),
                    f"{trip_index}_ori_country": "NDC",  # Valeur par défaut
                    f"{trip_index}_des_country": "NDC",  # Valeur par défaut
                    f"{trip_index}_governing_carrier": trip.get('OperatingCarrierInfo', {}).get('CarrierDesigCode', '')
                }
                all_legs_details.append(leg_details)

        # Fusionner tous les détails des legs dans un seul dictionnaire
        combined_leg_details = {}
        for leg_detail in all_legs_details:
            combined_leg_details.update(leg_detail)

        return combined_leg_details
    return {}

def concatenate_cities(row):
    number_of_legs = row['NB_LEGS']

    # Si le nombre de legs est supérieur à 3, renvoyer "check"
    if number_of_legs > 3:
        return "check"

    # Si le nombre de legs est inférieur ou égal à 2
    ori_city = row.get('0_ori_city', '')
    des_city = row.get('0_des_city', '')
    ori_country = row.get('0_ori_country', '')
    des_country = row.get('0_des_country', '')

    # Si 'NDC' est dans '0_ori_country', on change les variables pour utiliser les codes NDC
    if ori_country == 'NDC':
        ori_city = row.get('0_Ori_NDC', '')
        des_city = row.get('0_Des_NDC', '')
        ori_country = row.get('0_Ori_country_NDC', '')
        des_country = row.get('0_Des_country_NDC', '')

    # Maintenant, appliquer les règles en fonction des valeurs actuelles
    if 'PAR' in [ori_city, des_city]:
        if number_of_legs <= 2:
            return f"PAR {des_city}" if ori_city == 'PAR' else f"PAR {ori_city}"
        else:
            return f"PAR {des_city} !!!" if ori_city == 'PAR' else f"PAR {ori_city} !!!"
    elif ori_country == 'FR' and des_country == 'FR':
        if number_of_legs <= 2:
            return f"{sorted([ori_city, des_city])[0]} {sorted([ori_city, des_city])[1]}"
        else:
            return f"{sorted([ori_city, des_city])[0]} {sorted([ori_city, des_city])[1]} !!!"
    elif ori_country != 'FR' and des_country != 'FR':
        if number_of_legs <= 2:
            return f"{sorted([ori_city, des_city])[0]} {sorted([ori_city, des_city])[1]}"
        else:
            return f"{sorted([ori_city, des_city])[0]} {sorted([ori_city, des_city])[1]} !!!"
    elif ori_country == 'FR' and des_country != 'FR':
        if number_of_legs <= 2:
            return f"{ori_city} {des_city}"
        else:
            return f"{ori_city} {des_city} !!!"
    elif ori_country != 'FR' and des_country == 'FR':
        if number_of_legs <= 2:
            return f"{des_city} {ori_city}"
        else:
            return f"{des_city} {ori_city} !!!"

    return ""

def concatenate_carriers(row):
    carrier_columns = [col for col in row.index if col.endswith('_governing_carrier')]
    return ' '.join([str(row[col]) for col in carrier_columns if row[col]])

def classify_airline(concat_carrier):
    if any(carrier in concat_carrier for carrier in carrier_list):
        return "AIR FRANCE"
    return "INDUSTRIE"

def lookup_airport_details(concatenated_city):
    # Find the matching row where 'O&D restitué' matches the given concatenated_city
    match_row = airports_df[airports_df['O&D restitué'] == concatenated_city]

    if not match_row.empty:
        # Extract values for each field if a match is found
        haul_type = match_row['Haul type'].values[0]
        origin_area = match_row['Origin area'].values[0]
        annex_c = match_row['Annex C/ Out of Annex C 2023'].values[0]
        label_origin = match_row['Label cities of origin'].values[0]
        label_destination = match_row['Label cities of destination'].values[0]
        label_country_destination = match_row['Label country of destination'].values[0]  # Add this line

        # Return all the extracted values
        return haul_type, origin_area, annex_c, label_origin, label_destination, label_country_destination

    # Return empty strings if no match is found
    return "", "", "", "", "", ""

def lookup_ndc_code(city, primary_column, primary_code_column, secondary_column=None, secondary_code_column=None):
    # Chercher la ville dans le fichier CSV dans la colonne primaire (e.g., 'Airport origin code')
    code_row = airports_df[airports_df[primary_column] == city]

    if not code_row.empty:
        return code_row[primary_code_column].values[0]

    # Si aucune correspondance trouvée et les colonnes secondaires sont spécifiées
    if secondary_column and secondary_code_column:
        code_row = airports_df[airports_df[secondary_column] == city]
        if not code_row.empty:
            return code_row[secondary_code_column].values[0]

    return ""

def get_items_flight(id_soc, start_date):
    filter_criteria = {
        "society._id": ObjectId(id_soc),
        "createdAt": {"$gte": start_date},
        "type": "flight",
        "statusHistory.to": "confirmed",
        "status": {"$in": ["confirmed", "cancelled"]}
    }

    projection = {
        "id": 1,
        "offline": 1,
        "createdAt": 1,
        "type": 1,
        "status": 1,
        "billingId": 1,
        "travelers.billingId": 1,
        "detail.trips": 1,
        "_id": 0
    }

    items = col_items.find(filter_criteria, projection)
    data = []

    for item in items:
        offline_status = "Offline" if item.get("offline") else "Online"
        item_type = item.get("type")
        item_createdAt = item.get("createdAt")
        item_status = item.get("status", "Unknown")

        billing_id = item.get("billingId")
        if not billing_id:
            travelers = item.get("travelers", [])
            billing_id = next((traveler.get("billingId") for traveler in travelers if traveler.get("billingId")), None)

        raison = "Unknown"
        if billing_id:
            try:
                billing_doc = col_billings.find_one({"_id": ObjectId(billing_id)}, {"raison": 1})
            except errors.InvalidId:
                billing_doc = col_billings.find_one({"_id": billing_id}, {"raison": 1})

            if billing_doc:
                raison = billing_doc.get("raison", "Unknown")

        trips = item.get("detail", {}).get("trips", [])
        number_of_legs = sum([len(trip.get("legs", [])) if trip.get("legs", []) else 1 for trip in trips])

        basic_data = {
            "ITEM_ID": item.get("id"),
            "IS_OFFLINE": offline_status,
            "CREATED_AT": item_createdAt,
            "TYPE": item_type,
            "STATUS": item_status,
            "NB_LEGS": number_of_legs,
            "BILLING_ID": billing_id,
            "ENTITY": raison
        }

        leg_details = fetch_flight_details(item.get("id"))
        combined_data = {**basic_data, **leg_details}
        data.append(combined_data)

    df = pd.DataFrame(data)

    # Interroger le fichier CSV pour les colonnes avec 'NDC' dans les pays
    for i in range(2):  # Pour les index 0 et 1 (0_ori_country, 1_ori_country, etc.)
        ori_country_col = f'{i}_ori_country'
        des_country_col = f'{i}_des_country'
        ori_city_col = f'{i}_ori_city'
        des_city_col = f'{i}_des_city'

        # Si "NDC" dans ori_country, chercher dans 'Airport origin code', sinon dans 'Airport destination code'
        df[f'{i}_Ori_NDC'] = df.apply(
            lambda row: lookup_ndc_code(row[ori_city_col],
                                        'Airport origin code',
                                        'Origin Code',
                                        'Airport destination code',
                                        'Destination Code') if row[ori_country_col] == 'NDC' else '', axis=1)

        # Si "NDC" dans des_country, chercher dans 'Airport destination code', sinon dans 'Airport origin code'
        df[f'{i}_Des_NDC'] = df.apply(
            lambda row: lookup_ndc_code(row[des_city_col],
                                        'Airport destination code',
                                        'Destination Code',
                                        'Airport origin code',
                                        'Origin Code') if row[des_country_col] == 'NDC' else '', axis=1)

        # Condition pour Ori_country_NDC (Nouveau code)
        df[f'{i}_Ori_country_NDC'] = df.apply(
            lambda row: lookup_ndc_code(row[ori_city_col],
                                        'Airport origin code',
                                        'Country code of Origine',
                                        'Airport destination code',
                                        'Country code of destination') if row[ori_country_col] == 'NDC' else '', axis=1)

        # Condition pour Des_country_NDC (Nouveau code)
        df[f'{i}_Des_country_NDC'] = df.apply(
            lambda row: lookup_ndc_code(row[des_city_col],
                                        'Airport destination code',
                                        'Country code of destination',
                                        'Airport origin code',
                                        'Country code of Origine') if row[des_country_col] == 'NDC' else '', axis=1)

    # Appliquer la fonction concatenate_cities à chaque ligne du DataFrame
    df['CONCATENATED_CITIES'] = df.apply(concatenate_cities, axis=1)

    # Rechercher les détails supplémentaires dans 'airports_df' pour 'CONCATENATED_CITIES'
    df[['HAUL_TYPE', 'ORIGIN_AREA', 'ANNEX_C', 'LABEL_ORIGIN', 'LABEL_DESTINATION']] = df.apply(
        lambda row: pd.Series(lookup_airport_details(row['CONCATENATED_CITIES'])), axis=1)

    # Concaténer les colonnes qui terminent par 'governing_carrier' pour créer CONCAT_carrier
    df['CONCAT_carrier'] = df.apply(concatenate_carriers, axis=1)

    # Créer la colonne 'AIRLINE_GROUP' pour classifier en fonction de la présence d'un code de 'carrier_list'
    df['AIRLINE_GROUP'] = df['CONCAT_carrier'].apply(classify_airline)

    # Sauvegarder le DataFrame en CSV
    df.to_csv(f"csv/base/items_flight_{id_soc}.csv", index=False)

def merge_extract_flight(id_soc):
    df0 = pd.read_csv(f"csv/base/bills_{id_soc}.csv")
    df0 = df0[df0["type"] == "flight"]
    df1 = pd.read_csv(f"csv/base/items_flight_{id_soc}.csv")
    df = pd.merge(df0, df1, on='ITEM_ID', how='inner')

    # Convert CREATED_AT to YYYY-MM and YYYY formats
    df['ISSUED_MONTH'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%Y%m')
    df['ODLIST'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%Y')

    df.to_csv(f"csv/res/flight/merge_flight_{id_soc}.csv")


    columns_to_keep = [
        'ITEM_ID', 'type_bill','ISSUED_MONTH', 'ODLIST', 'IS_OFFLINE','TOTAL_BILLED',
        'NB_LEGS', 'CONCATENATED_CITIES', 'HAUL_TYPE', 'ORIGIN_AREA', 'ANNEX_C',
        'LABEL_ORIGIN', 'LABEL_DESTINATION', 'AIRLINE_GROUP',
    ]

    # Sélectionner uniquement les colonnes spécifiées
    df_filtered = df[columns_to_keep]

    # Affichage des premières lignes pour vérifier
    print(df_filtered.head())
    df_filtered = df_filtered.sort_values(by='ODLIST')

    # Sauvegarde du DataFrame filtré dans un fichier CSV
    df_filtered.to_csv(f'csv/res/flight/filtered_flight_{id_soc}.csv', index=False)

def group_flight(id_soc):
    # Charger le fichier CSV téléchargé
    file_path = f'csv/res/flight/filtered_flight_{id_soc}.csv'
    df = pd.read_csv(file_path)

    # Groupement par les colonnes spécifiées
    grouping_columns = [
        'CONCATENATED_CITIES', 'ORIGIN_AREA', 'ANNEX_C', 'LABEL_ORIGIN',
        'LABEL_DESTINATION', 'ISSUED_MONTH', 'ODLIST', 'IS_OFFLINE'
    ]

    # Agrégation des données
    df_grouped_af = df.groupby(grouping_columns).agg(
        TOTAL_BILLED_AIR_FRANCE=('TOTAL_BILLED', lambda x: x[df['AIRLINE_GROUP'] == 'AIR FRANCE'].sum()),
        TOTAL_BILLED_INDUSTRIE=('TOTAL_BILLED', lambda x: x[df['AIRLINE_GROUP'] == 'INDUSTRIE'].sum()),
        NB_LEGS_AIR_FRANCE=('NB_LEGS', lambda x: x[df['AIRLINE_GROUP'] == 'AIR FRANCE'].sum()),
        NB_LEGS_INDUSTRIE=('NB_LEGS', lambda x: x[df['AIRLINE_GROUP'] == 'INDUSTRIE'].sum())
    ).reset_index()

    # fichier brut
    df_grouped = df.groupby(grouping_columns).agg({
        'TOTAL_BILLED': 'sum',
        'NB_LEGS': 'sum'}).reset_index()

    df_grouped.to_csv(f"csv/res/flight/grouped_flight_{id_soc}.csv")

    # fichier af

    df_grouped_af[['CODE ORIGINE', 'CODE DESTINATION']] = df_grouped_af['CONCATENATED_CITIES'].str.split(' ', n=1, expand=True)

    # Renommer les colonnes
    df_grouped_af = df_grouped_af.rename(columns={
        "CONCATENATED_CITIES" : "O&D",
        "ISSUED_MONTH": "DATE D'EMISSION",
        'ORIGIN_AREA': 'ZONE ORIGINE',
        'ANNEX_C': 'PERIMETRE',
        'LABEL_ORIGIN': 'LIBELLE ORIGINE',
        'LABEL_DESTINATION': 'LIBELLE DESTINATION',
        'IS_OFFLINE': 'TYPE DE VENTE',
        'TOTAL_BILLED_AIR_FRANCE': 'CA GROUPE AF KL',
        'NB_LEGS_AIR_FRANCE': 'NB O&D GROUPE AF KL',
        'TOTAL_BILLED_INDUSTRIE': 'CA TOTAL INDUSTRIE',
        'NB_LEGS_INDUSTRIE': 'NB O&D INDUSTRIE',
    })
    df_grouped_af['CA GROUPE AF KL'] = df_grouped_af['CA GROUPE AF KL'].round(0).astype(int)
    df_grouped_af['NB O&D GROUPE AF KL'] = df_grouped_af['NB O&D GROUPE AF KL'].round(0).astype(int)
    df_grouped_af['CA TOTAL INDUSTRIE'] = df_grouped_af['CA TOTAL INDUSTRIE'].round(0).astype(int)
    df_grouped_af['NB O&D INDUSTRIE'] = df_grouped_af['NB O&D INDUSTRIE'].round(0).astype(int)
    df_grouped_af['REFERENCE AF'] = ''
    df_grouped_af['RAISON SOCIALE'] = ''
    df_grouped_af['IATA EMETTEUR'] = ''

    # Connexion en base pour recup OIN et NAME
    result = col_soc.find_one({"_id": ObjectId(id_soc)})

    # Extraire les variables "name" et "settings.flight.bluebizz"
    name = result.get("name", "Unknown")  # Assigner une valeur par défaut si name est None
    bluebizz = result.get("settings", {}).get("flight", {}).get("bluebizz", "Bluebizz")  # Valeur par défaut
    bluebizz_value = bluebizz if bluebizz else "Bluebizz"

    print(name, bluebizz_value)

    # Remplir les colonnes du DataFrame avec les valeurs récupérées
    df_grouped_af['REFERENCE AF'] = bluebizz_value
    df_grouped_af['RAISON SOCIALE'] = name

    # Trier les valeurs par "DATE D'EMISSION" et "O&D"
    df_grouped_af = df_grouped_af.sort_values(by=["DATE D'EMISSION", "O&D"], ascending=[False, True])

    # Vérifier si les valeurs de 'O&D' de flight_df sont présentes dans 'O&D restitué' de airports_df
    df_grouped_af['O&D_match'] = df_grouped_af['O&D'].isin(airports_df['O&D restitué'])

    # Filtrer les lignes où 'O&D' a une correspondance dans airports_df
    filtered_grouped_af = df_grouped_af[df_grouped_af['O&D_match']].drop(columns=['O&D_match'])

    # Sauvegarder le DataFrame filtré dans un nouveau fichier CSV
    filtered_grouped_af.to_csv(f"csv/res/flight/grouped_flight_{id_soc}.csv", index=False)

    print(f"File saved: csv/res/flight/grouped_flight_{id_soc}.csv")

################ 🚅 TRAIN 🚅 #####################

def fetch_train_details(item_id):
    # Chercher l'item dans la collection MongoDB
    item = col_items.find_one({"id": item_id})
    print(item_id)

    if item and item.get('type') == 'train':
        journeys = item.get('detail', {}).get('journeys', [])
        nombre_leg = len(journeys)  # Nombre de voyages/jambes

        all_legs_details = []

        # Boucle sur tous les voyages (journeys)
        for journey_index, journey in enumerate(journeys):
            # Vérifier si 'departure' et 'arrival' existent au niveau du voyage
            if journey.get('departure') and journey.get('arrival'):
                ori_locationId = journey['departure'].get('locationId', '')
                des_locationId = journey['arrival'].get('locationId', '')

                leg_details = {
                    f"{journey_index}_ori_name": journey['departure'].get('name', ''),
                    f"{journey_index}_des_name": journey['arrival'].get('name', ''),
                    f"{journey_index}_ori_locationId": ori_locationId,
                    f"{journey_index}_des_locationId": des_locationId,
                }
                all_legs_details.append(leg_details)
            else:
                # Si 'departure' et 'arrival' n'existent pas, chercher dans les 'segments'
                segments = journey.get('segments', [])
                for segment_index, segment in enumerate(segments):
                    ori_locationId = segment['departure'].get('locationId', '')
                    des_locationId = segment['arrival'].get('locationId', '')

                    leg_details = {
                        f"{segment_index}_ori_name": segment['departure'].get('name', ''),
                        f"{segment_index}_des_name": segment['arrival'].get('name', ''),
                        f"{segment_index}_ori_locationId": ori_locationId,
                        f"{segment_index}_des_locationId": des_locationId,
                    }
                    all_legs_details.append(leg_details)

        # Fusionner tous les détails des segments dans un seul dictionnaire
        combined_leg_details = {}
        for leg_detail in all_legs_details:
            combined_leg_details.update(leg_detail)

        return combined_leg_details, nombre_leg

    return {}, 0

def get_items_train(id_soc, start_date):
    filter_criteria = {
        "society._id": ObjectId(id_soc),
        "createdAt": {"$gte": start_date},
        "type": "train",
        "statusHistory.to": "confirmed",
        "status": {"$in": ["confirmed", "cancelled"]}
    }

    projection = {
        "id": 1,
        "offline": 1,
        "createdAt": 1,
        "type": 1,
        "status": 1,
        "billingId": 1,
        "travelers.billingId": 1,
        "detail.trips": 1,
        "_id": 0
    }

    items = col_items.find(filter_criteria, projection)
    data = []

    for item in items:
        offline_status = "Offline" if item.get("offline") else "Online"
        item_type = item.get("type")
        item_createdAt = item.get("createdAt")
        item_status = item.get("status", "Unknown")

        billing_id = item.get("billingId")
        if not billing_id:
            travelers = item.get("travelers", [])
            billing_id = next((traveler.get("billingId") for traveler in travelers if traveler.get("billingId")), None)

        raison = "Unknown"
        if billing_id:
            try:
                billing_doc = col_billings.find_one({"_id": ObjectId(billing_id)}, {"raison": 1})
            except errors.InvalidId:
                billing_doc = col_billings.find_one({"_id": billing_id}, {"raison": 1})

            if billing_doc:
                raison = billing_doc.get("raison", "Unknown")

        # Récupérer les détails des segments et le nombre de jambes (legs) avec `fetch_train_details`
        leg_details, nombre_leg = fetch_train_details(item.get("id"))

        # Initialisation de `basic_data`
        basic_data = {
            "ITEM_ID": item.get("id"),
            "IS_OFFLINE": offline_status,
            "CREATED_AT": item_createdAt,
            "TYPE": item_type,
            "STATUS": item_status,
            "NB_LEGS": nombre_leg,  # Utilisation de nombre_leg ici
            "BILLING_ID": billing_id,
            "ENTITY": raison
        }

        # Fusionner les données de base avec les détails des jambes
        combined_data = {**basic_data, **leg_details}

        # Ajout au DataFrame
        data.append(combined_data)

    # Créer le DataFrame
    df = pd.DataFrame(data)

    # Sauvegarder le DataFrame en CSV
    df.to_csv(f"csv/base/items_train_{id_soc}.csv", index=False)

    print(df.head())

def concat_and_remove_columns(id_soc):
    df = pd.read_csv(f"csv/base/items_train_{id_soc}.csv")
    ori_columns = [col for col in df.columns if col.startswith(tuple(f'{i}_ori_name' for i in range(2, 10)))]
    des_columns = [col for col in df.columns if col.startswith(tuple(f'{i}_des_name' for i in range(2, 10)))]
    ori_locationId_columns = [col for col in df.columns if col.startswith(tuple(f'{i}_ori_locationId' for i in range(2, 10)))]
    des_locationId_columns = [col for col in df.columns if col.startswith(tuple(f'{i}_des_locationId' for i in range(2, 10)))]

    # Concaténer les colonnes ori, des, locationId
    df['ori_concat'] = df[ori_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df['des_concat'] = df[des_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df['ori_locationId_concat'] = df[ori_locationId_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df['des_locationId_concat'] = df[des_locationId_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Supprimer les colonnes inutiles
    columns_to_remove = ori_columns + des_columns + ori_locationId_columns+ des_locationId_columns
    df.drop(columns=columns_to_remove, inplace=True)
    df.to_csv(f"csv/base/temp_{id_soc}.csv", index=False)

def normalize_station_names(id_soc):
    df = pd.read_csv(f"csv/base/temp_{id_soc}.csv")
    # Liste des colonnes à traiter
    name_columns = [col for col in df.columns if '_ori_name' in col or '_des_name' in col]

    # Dictionnaire des remplacements
    replacements = {
        "PARIS": "PARIS",
        "MONTPELLIER": "MONTPELLIER",
        "MARSEILLE": "MARSEILLE ST CHARLES",
        "BORDEAUX": "BORDEAUX",
        "STRASBOURG": "STRASBOURG",
        "LYON": "LYON"
    }

    # Fonction pour normaliser une entrée
    def normalize_name(name):
        if pd.isna(name):
            return name
        name_upper = name.upper()  # Mettre tout en majuscule pour ignorer la casse
        for keyword, replacement in replacements.items():
            if keyword in name_upper:
                return replacement
        return name  # Retourner le nom original s'il n'y a pas de correspondance

    # Appliquer la normalisation à toutes les colonnes pertinentes
    for col in name_columns:
        df[col] = df[col].apply(normalize_name)

    df.to_csv(f"csv/base/temp_{id_soc}.csv", index=False)

def fill_missing_location_ids(id_soc):
    # Charger les fichiers CSV
    temp_df = pd.read_csv(f"csv/base/temp_{id_soc}.csv")
    metropolis_df = pd.read_csv("const/O&D Metropolis rail 2024 v In&Out.csv", delimiter=";")
    european_df = pd.read_csv("const/O&D European Rail 2024 v In&Out.csv", delimiter=";")

    # Liste des colonnes à traiter
    location_columns = [
        '0_ori_locationId', '0_des_locationId', '1_ori_locationId', '1_des_locationId',
    ]

    # Liste des colonnes correspondantes pour les noms (origine et destination)
    name_columns = [
        '0_ori_name', '0_des_name', '1_ori_name', '1_des_name',
    ]

    # Fonction pour interroger les fichiers et remplir le locationId
    def lookup_location_id(name, ref_df):
        # Chercher dans 'Label of railway station of origin'
        origin_match = ref_df[ref_df['Label of railway station of origin'] == name]
        if not origin_match.empty:
            return origin_match['Railway station code NTS of origin'].values[0]
        else:
            # Chercher dans 'Label of railway station of destination'
            destination_match = ref_df[ref_df['Label of railway station of destination'] == name]
            if not destination_match.empty:
                return destination_match['Railway station code NTS of destination'].values[0]
        return "PAS TROUVÉ"

    # Remplir les valeurs manquantes pour chaque paire (locationId, name)
    for loc_col, name_col in zip(location_columns, name_columns):
        for idx, row in temp_df.iterrows():
            if pd.isna(row[loc_col]) and pd.notna(row[name_col]):
                # Chercher dans le fichier Metropolis
                result = lookup_location_id(row[name_col], metropolis_df)
                if result == "PAS TROUVÉ":
                    # Si pas trouvé dans Metropolis, chercher dans le fichier European
                    result = lookup_location_id(row[name_col], european_df)
                temp_df.at[idx, loc_col] = result
    temp_df = temp_df[(temp_df['0_ori_locationId'] != "PAS TROUVÉ") & (temp_df['0_des_locationId'] != "PAS TROUVÉ")]
    df = temp_df
    def determine_zone(row):
        # Vérifier si une des valeurs ne commence pas par 'FR'
        columns_to_check = ['0_ori_locationId', '0_des_locationId', '1_ori_locationId', '1_des_locationId']

        for col in columns_to_check:
            if pd.notna(row[col]) and not row[col].startswith("FR"):
                return "EUROPE"
        return "METRO"
    df['Zone'] = temp_df.apply(determine_zone, axis=1)

    # Créer la colonne 'Zone' en appliquant la fonction determine_zone à chaque ligne

    df.to_csv(f"csv/base/temp_{id_soc}.csv", index=False)

def process_locations(id_soc):
    # Charger le DataFrame
    df = pd.read_csv(f"csv/base/temp_{id_soc}.csv")
    rail_metro_df = pd.read_csv('const/O&D Metropolis rail 2024 v In&Out.csv', delimiter=";")
    rail_euro_df = pd.read_csv('const/O&D European Rail 2024 v In&Out.csv', delimiter=";")

    # Fonction pour déterminer les 'ori_location_rest' et 'des_location_rest'
    def determine_rest_locations(row):
        if row['0_ori_name'] == 'PARIS':
            return row['0_ori_locationId'], row['0_des_locationId']
        elif row['0_des_name'] == 'PARIS':
            return row['0_des_locationId'], row['0_ori_locationId']
        else:
            locations = sorted([row['0_ori_locationId'], row['0_des_locationId']])
            ori_name, des_name = sorted([row['0_ori_name'], row['0_des_name']])
            if ori_name == row['0_des_name']:
                return row['0_des_locationId'], row['0_ori_locationId']
            return locations[0], locations[1]

    # Créer les colonnes 'ori_location_rest' et 'des_location_rest'
    df['ori_location_rest'], df['des_location_rest'] = zip(*df.apply(determine_rest_locations, axis=1))

    # Fonction pour interroger rail_metro_df si la zone est 'METRO'
    def query_metropolis(ori_location, des_location):
        ori_code, ori_area, ori_label, des_code, des_label, des_country = None, None, None, None, None, None

        ori_match = rail_metro_df[rail_metro_df['Railway station code NTS of origin'] == ori_location]
        if not ori_match.empty:
            ori_code = ori_match['Origin code'].values[0]
            ori_area = ori_match['Origin area'].values[0]
            ori_label = ori_match['Label city of origin '].values[0]

        des_match = rail_metro_df[rail_metro_df['Railway station code NTS of destination'] == des_location]
        if not des_match.empty:
            des_code = des_match['Destination code'].values[0]
            des_country = des_match['Country code of destination'].values[0]
            des_label = des_match['Label city of destination '].values[0]

        return ori_code, ori_label, ori_area, des_code, des_label, des_country

    # Fonction pour interroger rail_euro_df si la zone est 'EUROPE'
    def query_europe(ori_location, des_location):
        ori_code, ori_area, ori_label, des_code, des_label, des_country = None, None, None, None, None, None

        ori_match = rail_euro_df[rail_euro_df['Railway station code NTS of origin'] == ori_location]
        if not ori_match.empty:
            ori_code = ori_match['Origin code'].values[0]
            ori_area = ori_match['Origin area'].values[0]
            ori_label = ori_match['Label city of origin '].values[0]

        des_match = rail_euro_df[rail_euro_df['Railway station code NTS of destination'] == des_location]
        if not des_match.empty:
            des_code = des_match['Destination code'].values[0]
            des_country = des_match['Country code of destination'].values[0]
            des_label = des_match['Label city of destination '].values[0]

        return ori_code, ori_label, ori_area, des_code, des_label, des_country

    # Appliquer les requêtes pour les lignes en fonction de la zone
    for idx, row in df.iterrows():
        if row['Zone'] == 'METRO':
            ori_code, ori_label, ori_area, des_code, des_label, des_country = query_metropolis(row['ori_location_rest'], row['des_location_rest'])
        elif row['Zone'] == 'EUROPE':
            ori_code, ori_label, ori_area, des_code, des_label, des_country = query_europe(row['ori_location_rest'], row['des_location_rest'])

        df.at[idx, 'ori_code'] = ori_code
        df.at[idx, 'ori_area'] = ori_area
        df.at[idx, 'des_code'] = des_code
        df.at[idx, 'des_country'] = des_country
        df.at[idx, 'ori_label'] = ori_label
        df.at[idx, 'des_label'] = des_label

    # Sauvegarder dans deux fichiers CSV
    df.to_csv(f"csv/base/items_train_{id_soc}.csv", index=False)

def merge_extract_train(id_soc):
    df0 = pd.read_csv(f"csv/base/bills_{id_soc}.csv")
    df0 = df0[df0["type"] == "train"]
    df1 = pd.read_csv(f"csv/base/items_train_{id_soc}.csv")
    df = pd.merge(df0, df1, on='ITEM_ID', how='inner')

    # Convert CREATED_AT to YYYY-MM and YYYY formats
    df['ISSUED_MONTH'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%Y%m')
    df['ODLIST'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%Y')
    df.to_csv(f"csv/res/train/merge_train_{id_soc}.csv")

def clean_train(id_soc):
    df = pd.read_csv(f'csv/res/train/merge_train_{id_soc}.csv')
    df['O&D'] = df['ori_code'].fillna('') + " " + df['des_code'].fillna('')
    grouped_columns = ['ISSUED_MONTH', 'ODLIST', 'O&D', 'ori_code', 'ori_label',
                       'des_code', 'des_label', 'des_country', 'Zone',"ori_area"]

    grouped_result = df.groupby(grouped_columns).agg(
        TOTAL_BILLED=('TOTAL_BILLED', 'sum'),
        NB_LEGS=('NB_LEGS', 'sum')
    ).reset_index()
    grouped_result = grouped_result.sort_values(by=['O&D', 'ISSUED_MONTH'], ascending=[True, False])

    # Connexion en base pour recup OIN et NAME
    result = col_soc.find_one({"_id": ObjectId(f"{id_soc}")},
                              {"name": 1, "settings.flight.bluebizz": 1, "_id": 0})

    # Extraire les variables "name" et "settings.flight.bluebizz"
    name = result.get("name", None)
    bluebizz = result.get("settings", {}).get("flight", {}).get("bluebizz", None)
    bluebizz_value = bluebizz if bluebizz else "Bluebizz"

    # Remplir les colonnes du DataFrame
    grouped_result['REFERENCE AF'] = bluebizz_value
    grouped_result['RAISON SOCIALE'] = name
    grouped_result['IATA EMETTEUR'] = ""

    #rename col
    grouped_result = grouped_result.rename(columns={
        "ISSUED_MONTH": "DATE D'EMISSION",
        'ori_area': 'ZONE ORIGINE',
        'ori_code': 'CODE ORIGINE',
        'des_code': 'CODE DESTINATION',
        'ori_label': 'LIBELLE ORIGINE',
        'des_label': 'LIBELLE DESTINATION',
        'des_country': 'PAYS DESTINATION',
        'TOTAL_BILLED': 'CA',
        'NB_LEGS': 'NB O&D',
        'des_country': 'PAYS DESTINATION',
    })
    columns_order = [
        "O&D","Zone",'RAISON SOCIALE','REFERENCE AF',"DATE D'EMISSION",'CODE ORIGINE','LIBELLE ORIGINE', 'CODE DESTINATION',"PAYS DESTINATION",'LIBELLE DESTINATION',"IATA EMETTEUR",'ZONE ORIGINE','CA','NB O&D'
    ]

    grouped_result = grouped_result[columns_order]
    grouped_result = grouped_result.sort_values(by=["DATE D'EMISSION","O&D"], ascending=[False, True])

    metro_df = grouped_result[grouped_result['Zone'] == 'METRO']
    europe_df = grouped_result[grouped_result['Zone'] == 'EUROPE']

    # Vérifier les correspondances pour la zone 'METRO'
    metro_df = metro_df[metro_df['Zone'] == 'METRO']
    metro_df['O&D_match'] = metro_df['O&D'].isin(rail_metro_df['O&D restitué'])

    # Filtrer les lignes où 'O&D' a une correspondance
    filtered_metro_df = metro_df[metro_df['O&D_match']].drop(columns=['O&D_match'])

    # Vérifier les correspondances pour la zone 'EUROPE'
    europe_df = europe_df[europe_df['Zone'] == 'EUROPE']
    europe_df['O&D_match'] = europe_df['O&D'].isin(rail_euro_df['O&D restitué'])

    # Filtrer les lignes où 'O&D' a une correspondance
    filtered_europe_df = europe_df[europe_df['O&D_match']].drop(columns=['O&D_match'])

    # Saving the two DataFrames to CSV files
    filtered_metro_df.to_csv(f'csv/res/train/grouped_train_metro_{id_soc}.csv')
    filtered_europe_df.to_csv(f'csv/res/train/grouped_train_euro_{id_soc}.csv')

################ 🤖 TRAITEMENT 🤖 #####################

def calculate_rr_flight(id_soc) :

    # Charger les données
    df = pd.read_csv(f"csv/res/flight/grouped_flight_{id_soc}.csv")

    # Convertir 'DATE D\'EMISSION' en datetime et extraire l'année et le mois
    df['DATE D\'EMISSION'] = pd.to_datetime(df['DATE D\'EMISSION'], format='%Y%m')
    df['YEAR'] = df['DATE D\'EMISSION'].dt.year
    df['MONTH'] = df['DATE D\'EMISSION'].dt.month

    # Filtrer les données pour "Online" uniquement
    df_online = df[df['TYPE DE VENTE'] == 'Online']

    # Grouper par O&D, DATE D'EMISSION
    df_grouped = df.groupby(['O&D', 'DATE D\'EMISSION']).agg({
        'CA TOTAL INDUSTRIE': 'sum',
        'CA GROUPE AF KL': 'sum',
        'NB O&D GROUPE AF KL': 'sum',
        'NB O&D INDUSTRIE': 'sum'
    }).reset_index()

    df_online_grouped = df_online.groupby(['O&D', 'DATE D\'EMISSION']).agg({
        'CA TOTAL INDUSTRIE': 'sum',
        'CA GROUPE AF KL': 'sum',
        'NB O&D GROUPE AF KL': 'sum',
        'NB O&D INDUSTRIE': 'sum'
    }).reset_index()

    # Extraire l'année et le mois après le groupement
    df_grouped['YEAR'] = df_grouped['DATE D\'EMISSION'].dt.year
    df_grouped['MONTH'] = df_grouped['DATE D\'EMISSION'].dt.month
    df_online_grouped['YEAR'] = df_online_grouped['DATE D\'EMISSION'].dt.year
    df_online_grouped['MONTH'] = df_online_grouped['DATE D\'EMISSION'].dt.month

    # Définir le mois et l'année précédents
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month

    # Si nous sommes en janvier, le mois précédent est décembre de l'année dernière
    if current_month == 1:
        prev_month = 12
        prev_year = current_year - 1
    else:
        prev_month = current_month - 1
        prev_year = current_year

    # Filtrer les données du mois N et N-1
    df_n = df_grouped[(df_grouped['YEAR'] == current_year) & (df_grouped['MONTH'] == prev_month)].drop_duplicates(subset=['O&D']).set_index('O&D')
    df_n.to_csv("test.csv")

    # Prendre les données du mois précédent de l'année précédente
    df_n_1 = df_grouped[(df_grouped['YEAR'] == current_year - 1) & (df_grouped['MONTH'] == prev_month)].drop_duplicates(
        subset=['O&D']).set_index('O&D')

    df_online_n = df_online_grouped[(df_online_grouped['YEAR'] == current_year) & (df_online_grouped['MONTH'] == prev_month)].drop_duplicates(subset=['O&D']).set_index('O&D')

    # Calcul des RR pour chaque colonne
    columns_to_calculate = ['CA TOTAL INDUSTRIE', 'CA GROUPE AF KL', 'NB O&D GROUPE AF KL', 'NB O&D INDUSTRIE']
    df_rr_combined = pd.DataFrame(index=df_n.index)
    for column in columns_to_calculate:
        df_rr = df_n[[column]].join(df_n_1[[column]], lsuffix='_N', rsuffix='_N_1')
        df_rr[f'RR_{column}'] = (((df_rr[f'{column}_N'] / df_rr[f'{column}_N_1']) - 1) * 100).fillna(0).round(2)
        # Join N-1 and N values to df_rr_combined
        df_rr_combined = df_rr_combined.join(df_rr[[f'{column}_N', f'{column}_N_1', f'RR_{column}']])

    # Save the final CSV with N and N-1 values along with RR calculations
    # Calcul des ratios TP CA et TP OD
    df_rr_combined['TP_CA_AF_KL'] = (df_n['CA GROUPE AF KL'] / df_n['CA TOTAL INDUSTRIE']).round(2)
    df_rr_combined['TP_OD_AF_KL'] = (df_n['NB O&D GROUPE AF KL'] / df_n['NB O&D INDUSTRIE']).round(2)
    df_rr_combined['TP_CA_AF_KL_N_1'] = (df_n_1['CA GROUPE AF KL'] / df_n_1['CA TOTAL INDUSTRIE']).round(2)
    df_rr_combined['TP_OD_AF_KL_N_1'] = (df_n_1['NB O&D GROUPE AF KL'] / df_n_1['NB O&D INDUSTRIE']).round(2)

    # Calcul des nouvelles colonnes pour "Online"
    df_rr_combined['CA_AF_KL_ONLINE'] = df_online_n['CA GROUPE AF KL'].fillna(0)
    df_rr_combined['OD_AF_KL_ONLINE'] = df_online_n['NB O&D GROUPE AF KL'].fillna(0)
    df_rr_combined['CA_INDUSTRIE_ONLINE'] = df_online_n['CA TOTAL INDUSTRIE'].fillna(0)
    df_rr_combined['OD_INDUSTRIE_ONLINE'] = df_online_n['NB O&D INDUSTRIE'].fillna(0)

    # Calcul des ratios "Online"
    df_rr_combined['TP_CA_AF_KL_ONLINE'] = (df_online_n['CA GROUPE AF KL'] / df_online_n['CA TOTAL INDUSTRIE']).round(2)
    df_rr_combined['TP_OD_AF_KL_ONLINE'] = (df_online_n['NB O&D GROUPE AF KL'] / df_online_n['NB O&D INDUSTRIE']).round(2)

    # Calcul des évolutions TP CA et TP OD
    df_rr_combined['EVOL_TP_CA'] = (df_rr_combined['TP_CA_AF_KL'] - df_rr_combined['TP_CA_AF_KL_N_1']).round(2)
    df_rr_combined['EVOL_TP_OD'] = (df_rr_combined['TP_OD_AF_KL'] - df_rr_combined['TP_OD_AF_KL_N_1']).round(2)

    #### CUMULS

    # Calculer les cumuls de janvier jusqu'au mois dernier pour 'CA TOTAL INDUSTRIE', 'CA GROUPE AF KL', etc.
    df_cumul = df_grouped[(df_grouped['YEAR'] == current_year) & (df_grouped['MONTH'] <= prev_month)]
    # Grouper par 'O&D' pour calculer les cumuls, sans écraser les colonnes existantes
    df_cumul_grouped = df_cumul.groupby('O&D').agg({
        'CA TOTAL INDUSTRIE': 'sum',
        'CA GROUPE AF KL': 'sum',
        'NB O&D GROUPE AF KL': 'sum',
        'NB O&D INDUSTRIE': 'sum'
    }).reset_index()

    # Renommer les colonnes pour indiquer qu'il s'agit des cumuls
    df_cumul_grouped.rename(columns={
        'CA TOTAL INDUSTRIE': 'cumul_CA_TOTAL_INDUSTRIE',
        'CA GROUPE AF KL': 'cumul_CA_GROUPE_AF_KL',
        'NB O&D GROUPE AF KL': 'cumul_NB_O&D_GROUPE_AF_KL',
        'NB O&D INDUSTRIE': 'cumul_NB_O&D_INDUSTRIE'
    }, inplace=True)

    # Calcul des ratios TP_CA et TP_OD pour les cumuls
    df_cumul_grouped['cumul_TP_CA_AF_KL'] = (df_cumul_grouped['cumul_CA_GROUPE_AF_KL'] / df_cumul_grouped['cumul_CA_TOTAL_INDUSTRIE']).round(2)
    df_cumul_grouped['cumul_TP_OD_AF_KL'] = (df_cumul_grouped['cumul_NB_O&D_GROUPE_AF_KL'] / df_cumul_grouped['cumul_NB_O&D_INDUSTRIE']).round(2)

    # Calcul des cumuls RR pour 'CA TOTAL INDUSTRIE' et 'NB O&D INDUSTRIE'
    df_cumul_grouped['cumul_RR_CA TOTAL INDUSTRIE'] = (df_cumul_grouped['cumul_CA_TOTAL_INDUSTRIE'].pct_change() * 100).round(2)
    df_cumul_grouped['cumul_RR_NB O&D INDUSTRIE'] = (df_cumul_grouped['cumul_NB_O&D_INDUSTRIE'].pct_change() * 100).round(2)

    # Calcul des cumuls RR pour 'CA GROUPE AF KL' et 'NB O&D GROUPE AF KL'
    df_cumul_grouped['cumul_RR_CA GROUPE AF KL'] = (df_cumul_grouped['cumul_CA_GROUPE_AF_KL'].pct_change() * 100).round(2)
    df_cumul_grouped['cumul_RR_NB O&D GROUPE AF KL'] = (df_cumul_grouped['cumul_NB_O&D_GROUPE_AF_KL'].pct_change() * 100).round(2)

    # Calcul des cumuls TP_CA_AF_KL et TP_OD_AF_KL pour les transactions "Online"
    df_online_cumul = df_online_grouped[(df_online_grouped['YEAR'] == current_year) & (df_online_grouped['MONTH'] < current_month)]
    df_online_cumul_grouped = df_online_cumul.groupby('O&D').agg({
        'CA TOTAL INDUSTRIE': 'sum',
        'CA GROUPE AF KL': 'sum',
        'NB O&D GROUPE AF KL': 'sum',
        'NB O&D INDUSTRIE': 'sum'
    }).reset_index()

    df_online_cumul_grouped['cumul_TP_CA_AF_KL_ONLINE'] = (df_online_cumul_grouped['CA GROUPE AF KL'] / df_online_cumul_grouped['CA TOTAL INDUSTRIE']).round(2)
    df_online_cumul_grouped['cumul_TP_OD_AF_KL_ONLINE'] = (df_online_cumul_grouped['NB O&D GROUPE AF KL'] / df_online_cumul_grouped['NB O&D INDUSTRIE']).round(2)

    # Vérification de l'année en cours et du mois précédent
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month

    # Calcul du mois précédent
    if current_month == 1:
        prev_month = 12  # Si c'est janvier, on passe à décembre
        prev_year = current_year - 1  # Et on soustrait 1 année
    else:
        prev_month = current_month - 1
        prev_year = current_year  # Pas de changement d'année si on est après janvier

    # Vérification de l'année N-1 pour les cumuls
    cumul_prev_year = current_year - 1  # Année N-1 pour les cumuls de janvier à mois courant N-1

    # Filtrer les données du mois N (mois dernier) et N-1 (même mois, année précédente)
    df_n = df_grouped[(df_grouped['YEAR'] == current_year) & (df_grouped['MONTH'] == prev_month)]
    df_n_1 = df_grouped[(df_grouped['YEAR'] == prev_year) & (df_grouped['MONTH'] == prev_month)]

    # Filtrer les données pour les cumuls de janvier à mois courant de l'année N-1
    df_cumul_n_1 = df_grouped[(df_grouped['YEAR'] == cumul_prev_year) & (df_grouped['MONTH'] <= prev_month)]

    # Grouper les données pour calculer les cumuls par O&D pour N-1
    df_cumul_n_1_grouped = df_cumul_n_1.groupby('O&D').agg({
        'CA TOTAL INDUSTRIE': 'sum',
        'CA GROUPE AF KL': 'sum',
        'NB O&D GROUPE AF KL': 'sum',
        'NB O&D INDUSTRIE': 'sum'
    }).reset_index()

    # Calculer le cumul TP_CA_AF_KL et TP_OD_AF_KL par O&D pour l'année N-1
    df_cumul_grouped['cumul_TP_CA_AF_KL_N_1'] = df_cumul_grouped['O&D'].map(
        df_cumul_n_1_grouped.set_index('O&D').apply(lambda row: (row['CA GROUPE AF KL'] / row['CA TOTAL INDUSTRIE']).round(2) if row['CA TOTAL INDUSTRIE'] > 0 else 0,
            axis=1))

    df_cumul_grouped['cumul_TP_OD_AF_KL_N_1'] = df_cumul_grouped['O&D'].map(df_cumul_n_1_grouped.set_index('O&D').apply(lambda row: (row['NB O&D GROUPE AF KL'] / row['NB O&D INDUSTRIE']).round(2) if row['NB O&D INDUSTRIE'] > 0 else 0,
            axis=1))

    # Calcul des cumuls EVOL_TP_CA et EVOL_TP_OD en utilisant les cumuls
    df_cumul_grouped['cumul_EVOL_TP_CA'] = (df_cumul_grouped['cumul_TP_CA_AF_KL'] - df_cumul_grouped['cumul_TP_CA_AF_KL_N_1']).round(2)

    df_cumul_grouped['cumul_EVOL_TP_OD'] = (df_cumul_grouped['cumul_TP_OD_AF_KL'] - df_cumul_grouped['cumul_TP_OD_AF_KL_N_1']).round(2)
    # Joindre les résultats "Online" à df_cumul_grouped sans écraser les colonnes
    df_cumul_grouped = df_cumul_grouped.join(df_online_cumul_grouped.set_index('O&D')[['cumul_TP_CA_AF_KL_ONLINE', 'cumul_TP_OD_AF_KL_ONLINE']], on='O&D')

    # Filtrer les données pour l'année N-1
    df_cumul_n_1 = df_grouped[(df_grouped['YEAR'] == current_year - 1) & (df_grouped['MONTH'] <= prev_month)]

    # Grouper par 'O&D' pour calculer les cumuls pour N-1
    df_cumul_n_1_grouped = df_cumul_n_1.groupby('O&D').agg({
        'CA TOTAL INDUSTRIE': 'sum',
        'CA GROUPE AF KL': 'sum',
        'NB O&D GROUPE AF KL': 'sum',
        'NB O&D INDUSTRIE': 'sum'
    }).reset_index()

    # Renommer les colonnes pour indiquer qu'il s'agit des cumuls N-1
    df_cumul_n_1_grouped.rename(columns={
        'CA TOTAL INDUSTRIE': 'cumul_CA_TOTAL_INDUSTRIE_N_1',
        'CA GROUPE AF KL': 'cumul_CA_GROUPE_AF_KL_N_1',
        'NB O&D GROUPE AF KL': 'cumul_NB_O&D_GROUPE_AF_KL_N_1',
        'NB O&D INDUSTRIE': 'cumul_NB_O&D_INDUSTRIE_N_1'
    }, inplace=True)

    # Fusionner ces données avec les cumuls de l'année en cours
    df_cumul_grouped = df_cumul_grouped.merge(df_cumul_n_1_grouped.set_index('O&D'), on='O&D', how='left')

    # Fusionner les résultats de RR, TP et cumuls avec df_rr_combined sans écraser les colonnes
    df_final = df_rr_combined.join(df_cumul_grouped.set_index('O&D'), on='O&D')

    # Sauvegarder le fichier final avec RR, ratios et cumuls
    df_final.to_csv(f"csv/res/flight/grouped_flight_with_rr_{id_soc}.csv")

def merge_rr(id_soc):
    import pandas as pd

    # Load grouped_flight_rr and filtered_flight datasets
    grouped_flight_rr = pd.read_csv(f'csv/res/flight/grouped_flight_with_rr_{id_soc}.csv')

    # Initialize new columns for the required details
    grouped_flight_rr['HAUL_TYPE'] = None
    grouped_flight_rr['ORIGIN_AREA'] = None
    grouped_flight_rr['ANNEX_C'] = None
    grouped_flight_rr['LABEL_ORIGIN'] = None
    grouped_flight_rr['LABEL_DESTINATION'] = None
    grouped_flight_rr['COUNTRY_OF_DEST'] = None  # New column for 'Label country of destination'

    # Use .loc to match and assign all requested columns where O&D matches CONCATENATED_CITIES
    for idx in grouped_flight_rr.index:
        od_value = grouped_flight_rr.loc[idx, 'O&D']
        match = airports_df.loc[airports_df['O&D restitué'] == od_value]

        if not match.empty:
            grouped_flight_rr.loc[idx, 'HAUL_TYPE'] = match['Haul type'].values[0]
            grouped_flight_rr.loc[idx, 'ORIGIN_AREA'] = match['Origin area'].values[0]
            grouped_flight_rr.loc[idx, 'ANNEX_C'] = match['Annex C/ Out of Annex C 2023'].values[0]
            grouped_flight_rr.loc[idx, 'LABEL_ORIGIN'] = match['Label cities of origin'].values[0]
            grouped_flight_rr.loc[idx, 'LABEL_DESTINATION'] = match['Label cities of destination'].values[0]
            grouped_flight_rr.loc[idx, 'COUNTRY_OF_DEST'] = match['Country code of destination'].values[0]


    # Split 'O&D' into 'ORI' and 'DEST'
    grouped_flight_rr[['ORI', 'DEST']] = grouped_flight_rr['O&D'].str.split(n=1, expand=True)


    grouped_flight_rr.to_csv(f'csv/res/flight/grouped_flight_full_{id_soc}.csv', index=False)

def split_final(id_soc):
    import pandas as pd

    # Load the input CSV files
    df0 = pd.read_csv(f"csv/res/flight/grouped_flight_with_rr_{id_soc}.csv")
    df1 = pd.read_csv(f"csv/res/flight/grouped_flight_full_{id_soc}.csv")

    # Merge df0 and df1 based on the common columns
    df = pd.merge(df0, df1, how='inner', on=[col for col in df0.columns if col in df1.columns])
    df.to_csv("aggreg.csv")
    # Define the desired column order
    columns_order = [
        'O&D', 'ANNEX_C', 'ORI', 'DEST', 'LABEL_ORIGIN', 'LABEL_DESTINATION', 'COUNTRY_OF_DEST',
        'HAUL_TYPE', 'ORIGIN_AREA', 'CA TOTAL INDUSTRIE_N', 'RR_CA TOTAL INDUSTRIE',
        'NB O&D INDUSTRIE_N', 'RR_NB O&D INDUSTRIE', 'CA GROUPE AF KL_N', 'RR_CA GROUPE AF KL',
        'NB O&D GROUPE AF KL_N', 'RR_NB O&D GROUPE AF KL', 'TP_CA_AF_KL', 'TP_CA_AF_KL_ONLINE',
        'EVOL_TP_CA', 'TP_OD_AF_KL', 'TP_OD_AF_KL_ONLINE', 'EVOL_TP_OD', 'cumul_CA_TOTAL_INDUSTRIE',
        'cumul_RR_CA TOTAL INDUSTRIE', 'cumul_NB_O&D_INDUSTRIE', 'cumul_RR_NB O&D INDUSTRIE',
        'cumul_CA_GROUPE_AF_KL', 'cumul_RR_CA GROUPE AF KL', 'cumul_NB_O&D_GROUPE_AF_KL',
        'cumul_RR_NB O&D GROUPE AF KL', 'cumul_TP_CA_AF_KL', 'cumul_TP_CA_AF_KL_ONLINE',
        'cumul_EVOL_TP_CA', 'cumul_TP_OD_AF_KL', 'cumul_TP_OD_AF_KL_ONLINE', 'cumul_EVOL_TP_OD'
    ]

    # Split the DataFrame based on the 'ANNEX_C' column
    df_split = dict(tuple(df.groupby('ANNEX_C')))

    # Iterate through the split DataFrames and save each part as a CSV file locally, reordering columns
    for key, df_part in df_split.items():
        # Reorder columns based on the predefined order
        df_part = df_part.reindex(columns=columns_order)

        # Create the filename based on the 'ANNEX_C' value
        filename = f"csv/OK/{key}.csv"

        # Save the DataFrame part with totals to a CSV file locally
        df_part.to_csv(filename, index=False)

    print("Files saved successfully with reordered columns for each split file!")

def aggreg_flight(id_soc):
    df = pd.read_csv(f"csv/res/flight/grouped_flight_{id_soc}.csv")

    # Reorder the columns based on the order you provided
    columns_order = [
        'O&D', 'RAISON SOCIALE', 'REFERENCE AF', 'DATE D\'EMISSION',
        'CODE ORIGINE', 'LIBELLE ORIGINE', 'CODE DESTINATION', 'LIBELLE DESTINATION',
        'PERIMETRE', 'IATA EMETTEUR', 'TYPE DE VENTE', 'ZONE ORIGINE',
        'CA TOTAL INDUSTRIE', 'NB O&D INDUSTRIE', 'CA GROUPE AF KL', 'NB O&D GROUPE AF KL'
    ]
    # Reorder the DataFrame
    df = df[columns_order]
    df = df.sort_values(by=['O&D', 'DATE D\'EMISSION'], ascending = [True, False])

    # Save the result to a new CSV file
    df.to_csv("csv/OK/aggreg_flight.csv",index = False)

def aggreg_train(id_soc):
    df_euro = pd.read_csv(f"csv/res/train/grouped_train_euro_{id_soc}.csv")
    df_metro = pd.read_csv(f"csv/res/train/grouped_train_metro_{id_soc}.csv")


    # Define the desired column order
    columns_order = [
        'O&D', 'Zone', 'RAISON SOCIALE', 'REFERENCE AF', 'DATE D\'EMISSION',
        'CODE ORIGINE', 'LIBELLE ORIGINE', 'CODE DESTINATION', 'LIBELLE DESTINATION',
        'PAYS DESTINATION', 'IATA EMETTEUR', 'ZONE ORIGINE', 'CA', 'NB O&D'
    ]

    # Reorder the columns for df_euro
    df_euro = df_euro.reindex(columns=columns_order)

    # Reorder the columns for df_metro
    df_metro = df_metro.reindex(columns=columns_order)

    # Save the results to new CSV files
    df_euro.to_csv(f"csv/OK/aggreg_rail_euro.csv", index=False)
    df_metro.to_csv(f"csv/OK/aggreg_rail_metro.csv", index=False)

    print("Reorganization and saving completed for both df_euro and df_metro.")

import pandas as pd

import pandas as pd

import pandas as pd

def calculate_rr_train(id_soc, df_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(f'csv/res/train/grouped_train_{df_name}_{id_soc}.csv')

    # Check if the DataFrame is empty
    if df.empty:
        print(f"The {df_name} dataset is empty. No calculations can be performed.")
        return

    # Convert 'DATE D'EMISSION' to datetime using the format '%Y%m'
    df["DATE D'EMISSION"] = pd.to_datetime(df["DATE D'EMISSION"], format='%Y%m')

    # Define current year and month
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month

    # Determine previous month
    if current_month == 1:
        prev_month = 12
        prev_year = current_year - 1
    else:
        prev_month = current_month - 1
        prev_year = current_year

    # Group by the specified columns
    group_columns = ["O&D", "CODE ORIGINE", "CODE DESTINATION", "LIBELLE ORIGINE", "LIBELLE DESTINATION"]

    # Column name for O&D
    od_column = 'NB O&D'

    # For the previous month of this year (N)
    df_prev_month_n = df[(df["DATE D'EMISSION"].dt.year == current_year) &
                         (df["DATE D'EMISSION"].dt.month == prev_month)].groupby(group_columns).agg(
        CA_last_month_N=("CA", "sum"),
        OD_last_month_N=(od_column, "sum")
    ).reset_index()

    # For the previous month of last year (N-1)
    df_prev_month_n_1 = df[(df["DATE D'EMISSION"].dt.year == current_year - 1) &
                           (df["DATE D'EMISSION"].dt.month == prev_month)].groupby(group_columns).agg(
        CA_last_month_N_1=("CA", "sum"),
        OD_last_month_N_1=(od_column, "sum")
    ).reset_index()

    # For January to the previous month of this year (N)
    df_jan_to_prev_month_n = df[(df["DATE D'EMISSION"].dt.year == current_year) &
                                (df["DATE D'EMISSION"].dt.month <= prev_month)].groupby(group_columns).agg(
        CA_jan_to_last_month_N=("CA", "sum"),
        OD_jan_to_last_month_N=(od_column, "sum")
    ).reset_index()

    # For January to the previous month of last year (N-1)
    df_jan_to_prev_month_n_1 = df[(df["DATE D'EMISSION"].dt.year == current_year - 1) &
                                  (df["DATE D'EMISSION"].dt.month <= prev_month)].groupby(group_columns).agg(
        CA_jan_to_last_month_N_1=("CA", "sum"),
        OD_jan_to_last_month_N_1=(od_column, "sum")
    ).reset_index()

    # Merge all results into a single dataframe
    df_aggregated = pd.merge(df_prev_month_n, df_prev_month_n_1, on=group_columns, how='outer')
    df_aggregated = pd.merge(df_aggregated, df_jan_to_prev_month_n, on=group_columns, how='outer')
    df_aggregated = pd.merge(df_aggregated, df_jan_to_prev_month_n_1, on=group_columns, how='outer')

    # Round non-RR columns to integers
    cols_to_round = [
        'CA_last_month_N', 'OD_last_month_N',
        'CA_last_month_N_1', 'OD_last_month_N_1',
        'CA_jan_to_last_month_N', 'OD_jan_to_last_month_N',
        'CA_jan_to_last_month_N_1', 'OD_jan_to_last_month_N_1'
    ]
    df_aggregated[cols_to_round] = df_aggregated[cols_to_round].round(0)

    # Calculate the Rate of Change (RR) for CA and O&D
    df_aggregated['RR_CA'] = (((df_aggregated['CA_last_month_N'] / df_aggregated['CA_last_month_N_1']) - 1) * 100).round(2)
    df_aggregated['RR_OD'] = (((df_aggregated['OD_last_month_N'] / df_aggregated['OD_last_month_N_1']) - 1) * 100).round(2)

    # Calculate the Cumulative Rate of Change (RR CUMUL) for CA and O&D
    df_aggregated['RR_CA_CUMUL'] = (((df_aggregated['CA_jan_to_last_month_N'] / df_aggregated['CA_jan_to_last_month_N_1']) - 1) * 100).round(2)
    df_aggregated['RR_OD_CUMUL'] = (((df_aggregated['OD_jan_to_last_month_N'] / df_aggregated['OD_jan_to_last_month_N_1']) - 1) * 100).round(2)

    # Calculate the total values for each metric
    total_values = df_aggregated[cols_to_round].sum()

    # Calculate the Rate of Change (RR) using total values
    rr_od_last_month = ((total_values['OD_last_month_N'] / total_values['OD_last_month_N_1'] - 1) * 100).round(2)
    rr_ca_last_month = ((total_values['CA_last_month_N'] / total_values['CA_last_month_N_1'] - 1) * 100).round(2)
    rr_ca_jan_to_last_month = ((total_values['CA_jan_to_last_month_N'] / total_values['CA_jan_to_last_month_N_1'] - 1) * 100).round(2)
    rr_od_jan_to_last_month = ((total_values['OD_jan_to_last_month_N'] / total_values['OD_jan_to_last_month_N_1'] - 1) * 100).round(2)

    # Create a DataFrame for the total row
    total_row = pd.DataFrame([{
        'O&D': 'Total',
        'CODE ORIGINE': '',
        'CODE DESTINATION': '',
        'LIBELLE ORIGINE': '',
        'LIBELLE DESTINATION': '',
        'CA_last_month_N': total_values['CA_last_month_N'],
        'OD_last_month_N': total_values['OD_last_month_N'],
        'CA_last_month_N_1': total_values['CA_last_month_N_1'],
        'OD_last_month_N_1': total_values['OD_last_month_N_1'],
        'CA_jan_to_last_month_N': total_values['CA_jan_to_last_month_N'],
        'OD_jan_to_last_month_N': total_values['OD_jan_to_last_month_N'],
        'CA_jan_to_last_month_N_1': total_values['CA_jan_to_last_month_N_1'],
        'OD_jan_to_last_month_N_1': total_values['OD_jan_to_last_month_N_1'],
        'RR_CA': rr_ca_last_month,
        'RR_OD': rr_od_last_month,
        'RR_CA_CUMUL': rr_ca_jan_to_last_month,
        'RR_OD_CUMUL': rr_od_jan_to_last_month
    }])

    # Append the total row to the aggregated DataFrame
    df_aggregated = pd.concat([df_aggregated, total_row], ignore_index=True)

    # Save the aggregated DataFrame with the total row to CSV
    df_aggregated.to_csv(f'csv/OK/{df_name}.csv', index=False)



def total(id_soc):
    import pandas as pd

    # Load the CSV file
    file_path = f'csv/res/flight/grouped_flight_full_{id_soc}.csv'
    df = pd.read_csv(file_path)

    # Reorganize columns: move columns ending with "_N_1" to the end
    columns_to_move = [col for col in df.columns if col.endswith('_N_1')]
    columns_remaining = [col for col in df.columns if col not in columns_to_move]
    new_column_order = columns_remaining + columns_to_move

    # Reorganize the dataframe
    df_reorganized = df[new_column_order]

    # Calculate the totals by grouping by the column 'ANNEX_C'
    grouped_totals = df_reorganized.groupby('ANNEX_C').sum(numeric_only=True)

    # Calculate the subtotals RR for each ANNEX_C group
    grouped_totals['RR_CA GROUPE AF KL'] = (((grouped_totals['CA GROUPE AF KL_N'] / grouped_totals['CA GROUPE AF KL_N_1']) - 1) * 100).round(2)
    grouped_totals['RR_NB O&D GROUPE AF KL'] = (((grouped_totals['NB O&D GROUPE AF KL_N'] / grouped_totals['NB O&D GROUPE AF KL_N_1']) - 1) * 100).round(2)
    grouped_totals['RR_CA TOTAL INDUSTRIE'] = (((grouped_totals['CA TOTAL INDUSTRIE_N'] / grouped_totals['CA TOTAL INDUSTRIE_N_1']) - 1) * 100).round(2)
    grouped_totals['RR_NB O&D INDUSTRIE'] = (((grouped_totals['NB O&D INDUSTRIE_N'] / grouped_totals['NB O&D INDUSTRIE_N_1']) - 1) * 100).round(2)

    # Calculate TP ratios for each ANNEX_C group
    grouped_totals['TP_CA_AF_KL'] = (grouped_totals['CA GROUPE AF KL_N'] / grouped_totals['CA TOTAL INDUSTRIE_N']).round(2)
    grouped_totals['TP_CA_AF_KL_N_1'] = (grouped_totals['CA GROUPE AF KL_N_1'] / grouped_totals['CA TOTAL INDUSTRIE_N_1']).round(2)
    grouped_totals['TP_OD_AF_KL'] = (grouped_totals['NB O&D GROUPE AF KL_N'] / grouped_totals['NB O&D INDUSTRIE_N']).round(2)
    grouped_totals['TP_OD_AF_KL_N_1'] = (grouped_totals['NB O&D GROUPE AF KL_N_1'] / grouped_totals['NB O&D INDUSTRIE_N_1']).round(2)

    # Calculations for online sales (hypothetical logic)
    grouped_totals['TP_CA_AF_KL_ONLINE'] = (grouped_totals['CA_AF_KL_ONLINE'] / grouped_totals['CA_INDUSTRIE_ONLINE']).round(2)
    grouped_totals['TP_OD_AF_KL_ONLINE'] = (grouped_totals['OD_AF_KL_ONLINE'] / grouped_totals['OD_INDUSTRIE_ONLINE']).round(2)

    # Evolution calculation
    grouped_totals['EVOL_TP_CA'] = (grouped_totals['TP_CA_AF_KL'] - grouped_totals['TP_CA_AF_KL_N_1']).round(2)
    grouped_totals['EVOL_TP_OD'] = (grouped_totals['TP_OD_AF_KL'] - grouped_totals['TP_OD_AF_KL_N_1']).round(2)

    # -----------------------------------
    # Cumulative Calculations (Prefix: 'cumul_')
    # -----------------------------------

    # RR Calculations for cumulative columns
    grouped_totals['cumul_RR_CA GROUPE AF KL'] = (((grouped_totals['cumul_CA_GROUPE_AF_KL'] / grouped_totals['cumul_CA_GROUPE_AF_KL_N_1']) - 1) * 100).round(2)
    grouped_totals['cumul_RR_NB O&D GROUPE AF KL'] = (((grouped_totals['cumul_NB_O&D_GROUPE_AF_KL'] / grouped_totals['cumul_NB_O&D_GROUPE_AF_KL_N_1']) - 1) * 100).round(2)
    grouped_totals['cumul_RR_CA TOTAL INDUSTRIE'] = (((grouped_totals['cumul_CA_TOTAL_INDUSTRIE'] / grouped_totals['cumul_CA_TOTAL_INDUSTRIE_N_1']) - 1) * 100).round(2)
    grouped_totals['cumul_RR_NB O&D INDUSTRIE'] = (((grouped_totals['cumul_NB_O&D_INDUSTRIE'] / grouped_totals['cumul_NB_O&D_INDUSTRIE_N_1']) - 1) * 100).round(2)

    # TP Ratios for cumulative columns
    grouped_totals['cumul_TP_CA_AF_KL'] = (grouped_totals['cumul_CA_GROUPE_AF_KL'] / grouped_totals['cumul_CA_TOTAL_INDUSTRIE']).round(2)
    grouped_totals['cumul_TP_CA_AF_KL_N_1'] = (grouped_totals['cumul_CA_GROUPE_AF_KL_N_1'] / grouped_totals['cumul_CA_TOTAL_INDUSTRIE_N_1']).round(2)
    grouped_totals['cumul_TP_OD_AF_KL'] = (grouped_totals['cumul_NB_O&D_GROUPE_AF_KL'] / grouped_totals['cumul_NB_O&D_INDUSTRIE']).round(2)
    grouped_totals['cumul_TP_OD_AF_KL_N_1'] = (grouped_totals['cumul_NB_O&D_GROUPE_AF_KL_N_1'] / grouped_totals['cumul_NB_O&D_INDUSTRIE_N_1']).round(2)

    # Calculations for cumulative online sales (hypothetical logic)
    grouped_totals['cumul_TP_CA_AF_KL_ONLINE'] = ((grouped_totals['cumul_CA_GROUPE_AF_KL'] / grouped_totals['cumul_CA_TOTAL_INDUSTRIE']) * 100).round(2)
    grouped_totals['cumul_TP_OD_AF_KL_ONLINE'] = ((grouped_totals['cumul_NB_O&D_GROUPE_AF_KL'] / grouped_totals['cumul_NB_O&D_INDUSTRIE']) * 100).round(2)

    # Evolution calculation for cumulative values
    grouped_totals['cumul_EVOL_TP_CA'] = (grouped_totals['cumul_TP_CA_AF_KL'] - grouped_totals['cumul_TP_CA_AF_KL_N_1']).round(2)
    grouped_totals['cumul_EVOL_TP_OD'] = (grouped_totals['cumul_TP_OD_AF_KL'] - grouped_totals['cumul_TP_OD_AF_KL_N_1']).round(2)

    # Save the results in a new CSV file
    output_path = 'csv/res/grouped_totals_flight.csv'
    grouped_totals.to_csv(output_path)

    columns_to_retain = [
        'CA TOTAL INDUSTRIE_N', 'RR_CA TOTAL INDUSTRIE',
        'NB O&D INDUSTRIE_N', 'RR_NB O&D INDUSTRIE', 'CA GROUPE AF KL_N',
        'RR_CA GROUPE AF KL', 'NB O&D GROUPE AF KL_N', 'RR_NB O&D GROUPE AF KL',
        'TP_CA_AF_KL', 'TP_CA_AF_KL_ONLINE', 'EVOL_TP_CA', 'TP_OD_AF_KL',
        'TP_OD_AF_KL_ONLINE', 'EVOL_TP_OD', 'cumul_CA_TOTAL_INDUSTRIE',
        'cumul_RR_CA TOTAL INDUSTRIE', 'cumul_NB_O&D_INDUSTRIE',
        'cumul_RR_NB O&D INDUSTRIE', 'cumul_CA_GROUPE_AF_KL',
        'cumul_RR_CA GROUPE AF KL', 'cumul_NB_O&D_GROUPE_AF_KL',
        'cumul_RR_NB O&D GROUPE AF KL', 'cumul_TP_CA_AF_KL',
        'cumul_TP_CA_AF_KL_ONLINE', 'cumul_EVOL_TP_CA', 'cumul_TP_OD_AF_KL',
        'cumul_TP_OD_AF_KL_ONLINE', 'cumul_EVOL_TP_OD'
    ]

    # Reorder and retain only the specified columns
    grouped_totals = grouped_totals[columns_to_retain]

    # Save the results in a new CSV file
    output_path = 'csv/OK/totals_flight.csv'
    grouped_totals.to_csv(output_path)

    print(f"Total calculations saved to {output_path}")

def merge_total_flight(id_soc):
    # Charger les données de `totals_flight.csv`
    totals_flight_path = 'csv/OK/totals_flight.csv'
    totals_flight = pd.read_csv(totals_flight_path)

    # Charger le fichier CSV approprié en fonction de la valeur de `ANNEX_C`
    for annex_c_value in totals_flight['ANNEX_C'].unique():
        # Charger le fichier correspondant à la valeur de ANNEX_C
        file_name = f"csv/OK/{annex_c_value}.csv"
        df_annex = pd.read_csv(file_name)

        # Filtrer la ligne dans totals_flight pour cette valeur de 'ANNEX_C'
        row_to_append = totals_flight[totals_flight['ANNEX_C'] == annex_c_value]

        if not row_to_append.empty:
            # Garder uniquement les colonnes à partir de 'ORIGIN_AREA'
            try:
                start_column_index = df_annex.columns.get_loc('ORIGIN_AREA')
            except KeyError:
                print(f"'ORIGIN_AREA' non trouvé dans les colonnes de {file_name}")
                continue

            columns_to_append = df_annex.columns[start_column_index:]

            # Trouver l'intersection des colonnes à ajouter avec les colonnes disponibles dans row_to_append
            columns_to_append = [col for col in columns_to_append if col in row_to_append.columns]

            if not columns_to_append:
                print(f"Aucune colonne correspondante trouvée pour ANNEX_C {annex_c_value} dans totals_flight")
                continue

            # S'assurer que seules les colonnes appropriées sont utilisées pour l'ajout
            row_to_append = row_to_append[columns_to_append]

            # Ajouter la ligne filtrée au DataFrame
            df_annex = pd.concat([df_annex, row_to_append], ignore_index=True)

            # Sauvegarder le DataFrame mis à jour dans le fichier CSV
            df_annex.to_csv(file_name, index=False)

            print(f"Ligne ajoutée au fichier {file_name}")

