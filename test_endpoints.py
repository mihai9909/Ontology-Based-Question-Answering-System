import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request
from app_utilities.graphdb_query import GraphdbQuery
from app import app

class QueryEndpointTest(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

    @patch('app_utilities.graphdb_query.GraphdbQuery.query')
    def test_query_success(self, mock_query):
        mock_response = MagicMock()
        mock_response.text = 'Mocked response text'
        mock_query.return_value = mock_response
        payload = {'query': 'SELECT * WHERE { ?s ?p ?o }'}
        headers = {'Content-Type': 'application/json'}
        response = self.client.get('/query', json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, 'Mocked response text')

    def test_query_no_query(self):
        response = self.client.get('/query', json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {'error': 'No query provided'})

if __name__ == '__main__':
    unittest.main()
