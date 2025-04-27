DROP DATABASE IF EXISTS Innovius;
CREATE DATABASE Innovius;
USE Innovius;

CREATE TABLE locations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(500) NOT NULL
);

INSERT INTO locations (name, address) VALUES 
('Netherlands', 'Amsterdam'),
('China', 'Beijing'),
('Saudi Arabia', 'Stockholm'),
('Sweden', 'Washington, D.C, Texas'),
('USA', 'London'),
('UK', 'Germany- Berlin and Hamburg'),
('Germany', 'Vietnam-Hanoi'),
('Vietnam', 'Canada-Ottawa and Belledune'),
('Australia', 'Spain-Madrid'),
('Oman', 'Namibia - Windhoek and Luderitz'),
('France', 'South Africa - Pretoria, Bloemfontein, and Cape Town(three capitals)'),
('Canada', 'Saudi Arabia - Riyadh'),
('Spain', 'Australia - Canberra'),
('India', 'Oman - Muscat'),
('Austria', 'France - Paris'),
('Taiwan', 'India - New Delhi,Mumbai and Hyderabad'),
('Japan', 'Austria - Vienna and Linz'),
('South Korea' , 'Taiwan - Taipei'),
('Morocco', 'Japan - Tokyo'),
('Namibia', 'South Korea - Seoul'),
('United Arab Emirates', 'Morocco - Rabat and Dakhla'),
('Egypt', 'United Arab Emirates - Abu Dhabi'),
('Kazakhstan', 'Egypt - Cairo and Ain Sokhna'),
('Chile', 'Kazakhstan - Astana(formerly Nur-Sultan)'),
('South Africa', 'Chile - Santiago and Punta Arenas');


