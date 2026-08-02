
import csv
#   - Word counter from a `.txt` file

def count_words_from_txt(file_path):
    """Count total words in a .txt file.

    :param file_path: 

    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        print(f"Total words in '{file_path}': {len(words)}")
    except FileNotFoundError:
        print("Error: .txt file not found.")
    except Exception as e:
        print("Error reading .txt file:", e)

#  - CSV reader → average a column

def average_csv_column(file_path, column_name):
    """Read a .csv file and average values in a numeric column.

    :param file_path: param column_name:
    :param column_name: 

    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            values = []
            for row in reader:
                try:
                    values.append(float(row[column_name]))
                except ValueError:
                    pass  # Skip non-numeric or missing values

        if values:
            avg = sum(values) / len(values)
            print(f"Average of '{column_name}' in '{file_path}': {avg:.2f}")
        else:
            print(f"No numeric data found in column '{column_name}'.")

    except FileNotFoundError:
        print("Error: .csv file not found.")
    except KeyError:
        print(f"Error: Column '{column_name}' not found in the CSV file.")
    except Exception as e:
        print("Error reading .csv file:", e)


def main():
    """ """
    print("=== Word Counter from .txt ===")
    txt_path = input("Enter path to .txt file: ").strip()
    count_words_from_txt(txt_path)

    print("\n=== CSV Column Averager ===")
    csv_path = input("Enter path to .csv file: ").strip()
    column_name = input("Enter column name to average: ").strip()
    average_csv_column(csv_path, column_name)


if __name__ == "__main__":
    main()

