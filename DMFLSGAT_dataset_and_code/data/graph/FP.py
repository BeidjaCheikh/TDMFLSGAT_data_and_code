import pandas as pd
import argparse

def convert_csv_to_smi(input_csv, output_smi):
    # Lire le CSV
    df = pd.read_csv(input_csv)

    # Vérification minimale
    if "smiles" not in df.columns:
        raise ValueError("ERROR: The CSV file must contain a 'smiles' column.")

    # Ouvrir le fichier .smi en écriture
    with open(output_smi, "w") as f:
        for i, row in df.iterrows():
            smiles = str(row["smiles"]).strip()

            # Créer un nom de molécule simple (mol1, mol2, ...)
            name = f"mol{i+1}"

            # Ajouter l'activité si présente
            if "ACTIVITY" in df.columns:
                name += f"_act{row['ACTIVITY']}"

            f.write(f"{smiles} {name}\n")

    print(f"Conversion completed successfully!")
    print(f"SMI file created: {output_smi}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to SMI for PaDEL-Descriptor")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output SMI file")
    args = parser.parse_args()

    convert_csv_to_smi(args.input, args.output)

