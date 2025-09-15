#!/usr/bin/env python3
"""
Script de test pour les endpoints de l'API Wellix
"""

import asyncio
import httpx
import json
from typing import Dict, Any

BASE_URL = "https://potential-goldfish-jj5qwqx69xwhp75w-8000.app.github.dev"

class WellixAPITester:
    def __init__(self):
        self.base_url = BASE_URL
        self.access_token = None
        self.user_data = None
    
    async def test_health(self):
        """Test l'endpoint health"""
        print("🏥 Test de l'endpoint health...")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
    
    async def test_register(self):
        """Test l'enregistrement d'un utilisateur"""
        print("\n👤 Test d'enregistrement utilisateur...")
        user_data = {
            "email": "test@example.com",
            "password": "TestPassword123!",
            "first_name": "Test",
            "last_name": "User"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json=user_data,
                    timeout=30.0
                )
                print(f"Status: {response.status_code}")
                if response.status_code < 400:
                    data = response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                    if "access_token" in data:
                        self.access_token = data["access_token"]
                        self.user_data = data.get("user", {})
                        return True
                else:
                    print(f"Error: {response.text}")
                return False
            except Exception as e:
                print(f"Exception: {e}")
                return False
    
    async def test_login(self):
        """Test la connexion d'un utilisateur"""
        print("\n🔐 Test de connexion utilisateur...")
        login_data = {
            "email": "test@example.com",
            "password": "TestPassword123!"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json=login_data,
                    timeout=30.0
                )
                print(f"Status: {response.status_code}")
                if response.status_code < 400:
                    data = response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                    if "access_token" in data:
                        self.access_token = data["access_token"]
                        self.user_data = data.get("user", {})
                        return True
                else:
                    print(f"Error: {response.text}")
                return False
            except Exception as e:
                print(f"Exception: {e}")
                return False
    
    async def test_protected_endpoint(self):
        """Test un endpoint protégé avec le token"""
        if not self.access_token:
            print("❌ Pas de token disponible pour tester les endpoints protégés")
            return False
        
        print("\n🛡️  Test d'endpoint protégé...")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/health-profiles/",
                    headers=headers,
                    timeout=30.0
                )
                print(f"Status: {response.status_code}")
                if response.status_code < 400:
                    data = response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                    return True
                else:
                    print(f"Error: {response.text}")
                return False
            except Exception as e:
                print(f"Exception: {e}")
                return False

async def main():
    """Fonction principale de test"""
    tester = WellixAPITester()
    
    print("🚀 Début des tests API Wellix")
    print("=" * 50)
    
    # Test 1: Health check
    health_ok = await tester.test_health()
    
    # Test 2: Enregistrement
    register_ok = await tester.test_register()
    
    # Test 3: Connexion (si l'enregistrement a échoué)
    if not register_ok:
        login_ok = await tester.test_login()
    else:
        login_ok = True
    
    # Test 4: Endpoint protégé
    if login_ok or register_ok:
        protected_ok = await tester.test_protected_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 Résumé des tests:")
    print(f"✅ Health check: {'✓' if health_ok else '✗'}")
    print(f"✅ Enregistrement: {'✓' if register_ok else '✗'}")
    print(f"✅ Connexion: {'✓' if login_ok else '✗'}")
    if 'protected_ok' in locals():
        print(f"✅ Endpoint protégé: {'✓' if protected_ok else '✗'}")

if __name__ == "__main__":
    asyncio.run(main())