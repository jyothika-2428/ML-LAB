import pandas as pd
import numpy as np


# Function to load purchase data
def load_purchase_data(file_name, sheet_name):
    purchase_data = pd.read_excel(file_name, sheet_name=sheet_name)

    feature_matrix = purchase_data[
        ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    ].to_numpy()

    output_vector = purchase_data["Payment (Rs)"].to_numpy()

    return feature_matrix, output_vector


# Function to calculate the rank of X
def calculate_rank(feature_matrix):
    return np.linalg.matrix_rank(feature_matrix)


# Function to calculate product costs using pseudo-inverse
def calculate_product_costs(feature_matrix, output_vector):
    pseudo_inverse = np.linalg.pinv(feature_matrix)
    product_costs = pseudo_inverse @ output_vector
    return product_costs


# Main function
def main():

    file_name = "Lab Session Data.xlsx"
    sheet_name = "Purchase data"      # <-- Correct sheet name

    feature_matrix, output_vector = load_purchase_data(file_name, sheet_name)

    matrix_rank = calculate_rank(feature_matrix)

    product_costs = calculate_product_costs(feature_matrix, output_vector)

    print("Feature Matrix (X):")
    print(feature_matrix)

    print("\nOutput Vector (y):")
    print(output_vector)

    print("\nDimension of Vector Space:", feature_matrix.shape[1])

    print("\nNumber of vectors in Dataset:", feature_matrix.shape[0])

    print("\nRank of Feature Matrix:", matrix_rank)

    print("\nCost of each product:")
    print("Candy =", round(product_costs[0], 2))
    print("Mango =", round(product_costs[1], 2))
    print("Milk Packet =", round(product_costs[2], 2))


if __name__ == "__main__":
    main()