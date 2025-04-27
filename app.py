from flask import Flask, jsonify
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# Function to connect to MySQL
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Replace with your MySQL username
            password="Bala@2000",  # Replace with your MySQL password
            database="Innovius"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None

@app.route('/locations', methods=["GET"])
def get_locations():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Unable to connect to the database"}), 500

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM locations")
        locations = cursor.fetchall()
        print("Locations data:", locations)
    except Error as e:
        return jsonify({"error": f"Error executing query: {e}"}), 500
    finally:
        cursor.close()  # Always close the cursor
        conn.close()    # Always close the connection

    return jsonify(locations)  # Returns JSON response

if __name__ == '__main__':
    app.run(debug=True)

