import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
import os

def load_data(path, output_dir):
    """
    path: Chemin vers le fichier CSV d'entrée (contenant les SMILES et les labels)
    output_dir: Dossier où les fichiers de sortie seront enregistrés
    """
    # Lire les données du fichier CSV
    data = pd.read_csv(path)
    smiles = list(data['smiles'])

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configuration des différents types de fingerprints et fichiers de sortie
    fingerprint_configs = [
            ('TopologicalTorsion', 'topological_torsion_fingerprints.csv'),

        ]

    # Calculer et sauvegarder les empreintes pour chaque type de fingerprint
    for fingerprint_type, output_file in fingerprint_configs:
        fs = []  # Liste pour stocker les fingerprints

        # Parcourir chaque molécule (SMILES)
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)

            # Calcul des empreintes en fonction du type
            if fingerprint_type == 'Morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            elif fingerprint_type == 'ECFP':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            elif fingerprint_type == 'MACCS':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fingerprint_type == 'AtomPair':
                fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=4096)
            elif fingerprint_type == 'TopologicalTorsion':
                fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024)
            elif fingerprint_type == 'GraphBased':
                # Correction: Utilisation de GetMorganFingerprintAsBitVect
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            elif fingerprint_type == 'Pharmacophore':
                # Utilisation d'un motif SMARTS comme exemple de pharmacophore
                smarts = '[NX3][CX3](=[OX1])[#6]'  # Exemple: motif amide
                pharmacophore_fp = Chem.MolFromSmarts(smarts)
                fp = mol.HasSubstructMatch(pharmacophore_fp)
                fp = np.array([fp] * 1024)  # Ajuster la taille si nécessaire

            # Ajouter d'autres fingerprints ici si nécessaire

            # Convertir l'empreinte en tableau numpy et l'ajouter à la liste
            fs.append(np.array(fp))

        # Convertir la liste des fingerprints en un array numpy
        fingerprint_array = np.array(fs)
        df = pd.DataFrame(fingerprint_array)

        # Sauvegarder les résultats dans un fichier CSV
        output_filepath = os.path.join(output_dir, output_file)
        df.to_csv(output_filepath, index=False)

        print(f'{fingerprint_type} fingerprints saved to {output_filepath}')

# Exemple d'utilisation
path = r'/home/enset/Téléchargements/DMFGAM/Smiles.csv'
output_dir = r'/home/enset/Téléchargements/DMFGAM'

# Charger les données et calculer les empreintes
load_data(path, output_dir)
