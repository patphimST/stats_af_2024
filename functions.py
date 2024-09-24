from pymongo import MongoClient
import pandas as pd
import dns.resolver
import certifi
from bson import ObjectId, errors
import config
from datetime import datetime

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
                # Convertir les ObjectId en str
            }
            data.append(item_data)

    # Convertir la liste en DataFrame
    df = pd.DataFrame(data)

    df.to_csv(f"csv/base/bills_{id_soc}.csv")
    # Afficher le DataFrame
    print(df)

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
    match_row = airports_df[airports_df['O&D restitué'] == concatenated_city]

    if not match_row.empty:
        haul_type = match_row['Haul type'].values[0]
        origin_area = match_row['Origin area'].values[0]
        annex_c = match_row['Annex C/ Out of Annex C 2023'].values[0]
        label_origin = match_row['Label cities of origin'].values[0]
        label_destination = match_row['Label cities of destination'].values[0]
        return haul_type, origin_area, annex_c, label_origin, label_destination

    return "", "", "", "", ""  # Si aucune correspondance n'est trouvée

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
def get_items(id_soc, start_date):
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
    df.to_csv(f"csv/base/items_{id_soc}.csv", index=False)
    print(df.head())  # Afficher les premières lignes du DataFrame

def merge_extract(id_soc):
    df0 = pd.read_csv(f"csv/base/bills_{id_soc}.csv")
    df0 = df0[df0["type"] == "flight"]
    df1 = pd.read_csv(f"csv/base/items_{id_soc}.csv")
    df = pd.merge(df0, df1, on='ITEM_ID', how='inner')

    # Convert CREATED_AT to YYYY-MM and YYYY formats
    df['ISSUED_MONTH'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%Y%m')
    df['ODLIST'] = pd.to_datetime(df['CREATED_AT']).dt.strftime('%Y')

    df.to_csv(f"csv/res//merge_{id_soc}.csv")


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
    df_filtered.to_csv(f'csv/res/filtered_{id_soc}.csv', index=False)


