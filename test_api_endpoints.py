#!/usr/bin/env python3
"""
Script de test pour les endpoints de l'API Wellix
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "https://potential-goldfish-jj5qwqx69xwhp75w-8000.app.github.dev"

class APITester:
    def __init__(self):
        self.base_url = BASE_URL
        self.access_token = None
        self.user_id = None
        
    async def test_health(self):
        """Test endpoint de santé"""
        print("🏥 Test endpoint /health")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.json()}")
                return response.status_code == 200
            except Exception as e:
                print(f"❌ Erreur: {e}")
                return False
    
    async def test_register(self):
        """Test inscription utilisateur"""
        print("\n👤 Test endpoint /api/v1/auth/register")
        
        user_data = {
            "email": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json=user_data,
                    headers={"Content-Type": "application/json"}
                )
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                
                if response.status_code == 201:
                    data = response.json()
                    self.access_token = data.get("access_token")
                    self.user_id = data.get("user", {}).get("id")
                    print(f"✅ Inscription réussie - Token: {self.access_token[:20]}...")
                    return True
                else:
                    print(f"❌ Inscription échouée: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Erreur: {e}")
                return False
    
    async def test_login(self):
        """Test connexion utilisateur"""
        print("\n🔐 Test endpoint /api/v1/auth/login")
        
        # Utilisons des credentials de test connus
        login_data = {
            "email": "admin@wellix.com",
            "password": "admin123"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json=login_data,
                    headers={"Content-Type": "application/json"}
                )
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    self.access_token = data.get("access_token")
                    print(f"✅ Connexion réussie - Token: {self.access_token[:20]}...")
                    return True
                else:
                    print(f"❌ Connexion échouée: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Erreur: {e}")
                return False
    
    async def test_protected_endpoint(self):
        """Test endpoint protégé avec token"""
        print("\n🔒 Test endpoint protégé /api/v1/health-profiles/")
        
        if not self.access_token:
            print("❌ Aucun token disponible")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/health-profiles/",
                    headers=headers
                )
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                
                if response.status_code == 200:
                    print("✅ Accès autorisé avec token")
                    return True
                else:
                    print(f"❌ Accès refusé: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Erreur: {e}")
                return False
    
    async def test_health_profile_crud(self):
        """Test CRUD profils de santé"""
        print("\n💊 Test CRUD profils de santé")
        
        if not self.access_token:
            print("❌ Aucun token disponible")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Créer un profil de santé
        profile_data = {
            "conditions": ["diabetes"],
            "dietary_restrictions": ["gluten_free"],
            "allergies": ["nuts"],
            "fitness_goals": ["weight_loss"],
            "medical_notes": "Test profile for API testing"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # CREATE
                response = await client.post(
                    f"{self.base_url}/api/v1/health-profiles/",
                    json=profile_data,
                    headers=headers
                )
                print(f"CREATE Status: {response.status_code}")
                print(f"CREATE Response: {response.text}")
                
                if response.status_code == 201:
                    profile_id = response.json().get("id")
                    print(f"✅ Profil créé avec ID: {profile_id}")
                    
                    # READ
                    response = await client.get(
                        f"{self.base_url}/api/v1/health-profiles/{profile_id}",
                        headers=headers
                    )
                    print(f"READ Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        print("✅ Profil lu avec succès")
                        return True
                
                return False
                    
            except Exception as e:
                print(f"❌ Erreur: {e}")
                return False
    
    async def run_all_tests(self):
        """Exécute tous les tests"""
        print("🚀 Démarrage des tests API Wellix")
        print("=" * 50)
        
        results = {}
        
        # Test de base
        results["health"] = await self.test_health()
        
        # Tests d'authentification
        results["register"] = await self.test_register()
        if not results["register"]:
            # Si l'inscription échoue, essayons la connexion
            results["login"] = await self.test_login()
        
        # Tests endpoints protégés
        if self.access_token:
            results["protected"] = await self.test_protected_endpoint()
            results["health_profiles"] = await self.test_health_profile_crud()
        
        # Résumé
        print("\n" + "=" * 50)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 50)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        print(f"\nRésultat: {passed_tests}/{total_tests} tests réussis")
        
        return results

async def main():
    tester = APITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())