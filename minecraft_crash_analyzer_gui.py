#!/usr/bin/env python3
"""
마인크래프트 크래시 분석기 - Minecraft Crash Analyzer
마인크래프트 게임이 튕겼을 때 로그를 분석해서 쉽게 설명해줍니다.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import re
from pathlib import Path
import threading
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np


class AIErrorClassifier:
    """scikit-learn 기반 마인크래프트 AI 오류 분류기"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, lowercase=True, stop_words='english')
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.error_categories = []
        self.model_path = "minecraft_error_model.pkl"
        self.vectorizer_path = "minecraft_vectorizer.pkl"
        self._load_model()
    
    def _load_model(self):
        """저장된 모델 로드"""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.is_trained = True
            except:
                self._train_default_model()
        else:
            self._train_default_model()
    
    def _train_default_model(self):
        """기본 학습 데이터로 모델 학습"""
        training_data = [
            # 메모리 부족
            "OutOfMemoryError Java heap space",
            "GC overhead limit exceeded",
            "Cannot allocate memory",
            "Memory allocation failed",
            
            # 모드 충돌
            "Mixin conflict detected",
            "Duplicate mod detected",
            "Already registered",
            "Incompatible mod version",
            
            # 그래픽 오류
            "OpenGL error",
            "GLFW error",
            "Graphics rendering failed",
            "Shader compilation error",
            
            # 자바 오류
            "NullPointerException",
            "ClassNotFoundException",
            "NoClassDefFoundError",
            "Exception in thread",
            
            # 월드 손상
            "Corrupted world data",
            "Invalid NBT data",
            "Region file corrupted",
            "Chunk data damaged",
            
            # 네트워크 오류
            "Connection refused",
            "Connection timeout",
            "Socket error",
            "Network unreachable",
            
            # 버전 불일치
            "Version mismatch",
            "Protocol version incompatible",
            "Outdated client",
            "Outdated server",
            
            # 파일 없음
            "FileNotFoundError",
            "Cannot find file",
            "No such file",
            "Missing required file",
            
            # 권한 오류
            "Permission denied",
            "Access denied",
            "Cannot write file",
            "Cannot read file",
            
            # 텍스처 오류
            "Missing texture",
            "Texture not found",
            "Resource pack error",
            "Invalid texture pack",
            
            # 런처 오류
            "Launcher error",
            "Login failed",
            "Authentication failed",
            "Account error",
            
            # 모드 로더 오류
            "Forge error",
            "Fabric loader error",
            "Bootstrap error",
            "Tweak class error",
            
            # 청크 로딩 오류
            "Chunk loading error",
            "Terrain generation failed",
            "Chunk data error",
            "Unloading chunk failed",
            
            # 반복 크래시
            "Crash loop detected",
            "Repeatedly crashing",
            "Infinite crash",
            "Keeps crashing",
            
            # 렉/프리징
            "Lag spike detected",
            "Game freeze",
            "Stutter detected",
            "Low performance",
            
            # 사운드 오류
            "Sound error",
            "Audio device error",
            "Speaker error",
            "Microphone error",
            
            # 컨트롤러 오류
            "Controller error",
            "Gamepad error",
            "Joystick error",
            "Input device error",
            
            # 스킨 오류
            "Skin download failed",
            "Profile error",
            "Texture download failed",
            "Cape error",
        ]
        
        labels = [
            0, 0, 0, 0,  # 메모리
            1, 1, 1, 1,  # 모드 충돌
            2, 2, 2, 2,  # 그래픽
            3, 3, 3, 3,  # 자바
            4, 4, 4, 4,  # 월드
            5, 5, 5, 5,  # 네트워크
            6, 6, 6, 6,  # 버전
            7, 7, 7, 7,  # 파일
            8, 8, 8, 8,  # 권한
            9, 9, 9, 9,  # 텍스처
            10, 10, 10, 10,  # 런처
            11, 11, 11, 11,  # 모드 로더
            12, 12, 12, 12,  # 청크
            13, 13, 13, 13,  # 반복 크래시
            14, 14, 14, 14,  # 렉
            15, 15, 15, 15,  # 사운드
            16, 16, 16, 16,  # 컨트롤러
            17, 17, 17, 17,  # 스킨
        ]
        
        self.error_categories = [
            'out_of_memory', 'mod_conflict', 'graphics_error', 'java_error',
            'world_corruption', 'network_error', 'version_mismatch', 'file_not_found',
            'permission_error', 'texture_error', 'launcher_error', 'mod_loader_error',
            'chunk_loading_error', 'crash_loop', 'lag_freeze', 'sound_error',
            'controller_error', 'skin_error'
        ]
        
        X = self.vectorizer.fit_transform(training_data)
        self.classifier.fit(X, labels)
        self.is_trained = True
        self._save_model()
    
    def _save_model(self):
        """모델 저장"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        except:
            pass
    
    def predict_error_type(self, text: str) -> tuple:
        """텍스트에서 오류 타입 예측
        
        Returns:
            (error_category, confidence) - 오류 카테고리와 신뢰도 (0-100)
        """
        if not self.is_trained or not text.strip():
            return None, 0
        
        try:
            X = self.vectorizer.transform([text])
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            confidence = int(max(probabilities) * 100)
            
            if confidence < 30:  # 신뢰도 30% 미만은 무시
                return None, 0
            
            return self.error_categories[prediction], confidence
        except:
            return None, 0


class MinecraftCrashAnalyzer:
    # 마인크래프트 특화 오류 패턴
    MINECRAFT_ERRORS = {
        'out_of_memory': {
            'pattern': r'(OutOfMemoryError|java\.lang\.OutOfMemoryError|heap space|GC overhead)',
            'title': '메모리 부족',
            'description': '마인크래프트에 할당된 메모리가 부족합니다. 게임이 더 이상 메모리를 할당받지 못해 튕깁니다.',
            'severity': 'CRITICAL',
            'solutions': [
                '1. 런처에서 할당 메모리를 늘려보세요 (최소 2GB, 권장 4GB)',
                '   - 런처 > 설정 > Java 설정 > JVM 인수',
                '   - -Xmx2G를 -Xmx4G로 변경 (4GB로 설정)',
                '2. 불필요한 프로그램을 종료하세요 (크롬, 디스코드 등)',
                '3. 월드의 청크를 정리해보세요',
                '4. 모드를 줄여보세요 (특히 그래픽 모드)',
                '5. 렌더 거리를 줄여보세요 (설정 > 비디오 설정 > 렌더 거리: 8-10)',
                '6. 파티클 효과를 줄여보세요 (설정 > 비디오 설정 > 파티클: 최소)'
            ]
        },
        'mod_conflict': {
            'pattern': r'(Mixin|mixin|conflict|duplicate|already registered|incompatible mod)',
            'title': '모드 충돌',
            'description': '설치된 모드들이 서로 충돌하고 있습니다. 같은 기능을 하는 모드가 여러 개 있거나 모드 간 호환성 문제가 있습니다.',
            'severity': 'HIGH',
            'solutions': [
                '1. 최근에 설치한 모드를 제거해보세요',
                '2. 모드 버전이 마인크래프트 버전과 맞는지 확인하세요',
                '   - 예: 1.20.1 버전 마인크래프트에는 1.20.1 모드만 설치',
                '3. 모드 의존성을 확인하세요 (필요한 라이브러리 모드)',
                '4. 모드 폴더를 백업하고 처음부터 설치해보세요',
                '5. 모드 로더 (Forge, Fabric) 버전을 확인하세요',
                '6. 모드 개발자 페이지에서 호환성 정보를 확인하세요'
            ]
        },
        'graphics_error': {
            'pattern': r'(OpenGL error|GLFW error|graphics error|render error|shader error|display error|GLX error)',
            'title': '그래픽 오류',
            'description': '그래픽 카드나 드라이버 문제입니다. 게임이 화면을 그리지 못해 튕기거나 화면이 이상하게 보입니다.',
            'severity': 'HIGH',
            'solutions': [
                '1. 그래픽 드라이버를 최신 버전으로 업데이트하세요',
                '   - NVIDIA: nvidia.com > 드라이버 다운로드',
                '   - AMD: amd.com > 드라이버 다운로드',
                '   - Intel: intel.com > 드라이버 다운로드',
                '2. 게임 설정에서 그래픽 옵션을 낮춰보세요',
                '   - 설정 > 비디오 설정 > 그래픽: 빠름',
                '3. 셰이더를 비활성화해보세요',
                '4. 마인크래프트를 재설치해보세요',
                '5. 그래픽 카드 드라이버를 완전히 제거하고 재설치해보세요'
            ]
        },
        'java_error': {
            'pattern': r'(java\.lang\.|NullPointerException|ClassNotFoundException|NoClassDefFoundError|Exception in thread)',
            'title': '자바 오류',
            'description': '자바 실행 중 오류가 발생했습니다. 자바 버전 문제이거나 필요한 파일이 손상되었을 수 있습니다.',
            'severity': 'HIGH',
            'solutions': [
                '1. 자바를 최신 버전으로 업데이트하세요',
                '   - java.com에서 최신 자바 다운로드',
                '2. 마인크래프트 런처를 재시작하세요',
                '3. 마인크래프트를 재설치해보세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 자바를 완전히 제거하고 재설치해보세요',
                '6. 런처 설정에서 자바 경로를 확인하세요'
            ]
        },
        'world_corruption': {
            'pattern': r'(corrupt|invalid|NBT|region file|chunk data|damaged world)',
            'title': '월드 손상',
            'description': '저장된 월드 파일이 손상되었습니다. 특정 위치에서만 튕기거나 월드를 로드할 수 없습니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 백업 폴더에서 이전 버전의 월드를 복구해보세요',
                '   - .minecraft/saves 폴더에서 월드 폴더 찾기',
                '2. 문제가 있는 청크를 삭제해보세요',
                '   - MCEdit 또는 WorldEdit 모드 사용',
                '3. 월드를 새로 만들어보세요',
                '4. 월드 백업 파일을 확인하세요',
                '5. 월드 파일을 다른 컴퓨터에서 열어보세요'
            ]
        },
        'network_error': {
            'pattern': r'(Connection refused|Connection timeout|socket error|network error|unreachable|Connection reset)',
            'title': '네트워크 오류',
            'description': '서버 연결에 문제가 있습니다. 인터넷 연결이 끊어졌거나 서버가 오프라인 상태입니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 인터넷 연결을 확인하세요',
                '2. 방화벽 설정을 확인하세요',
                '   - Windows Defender 방화벽에서 마인크래프트 허용',
                '3. 서버가 온라인 상태인지 확인하세요',
                '4. 라우터를 재부팅해보세요',
                '5. 서버 주소와 포트 번호를 다시 확인하세요',
                '6. VPN을 사용 중이면 비활성화해보세요'
            ]
        },
        'version_mismatch': {
            'pattern': r'(version mismatch|protocol version|incompatible version|outdated client|outdated server)',
            'title': '버전 불일치',
            'description': '클라이언트와 서버의 버전이 맞지 않습니다. 같은 버전을 사용해야 연결할 수 있습니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 마인크래프트 버전을 확인하세요',
                '   - 런처 > 게임 버전 확인',
                '2. 서버 버전과 클라이언트 버전을 맞추세요',
                '3. 모드 버전을 확인하세요',
                '4. 런처에서 올바른 버전을 선택했는지 확인하세요',
                '5. 서버 관리자에게 서버 버전을 물어보세요',
                '6. 모드 로더 버전도 맞는지 확인하세요'
            ]
        },
        'file_not_found': {
            'pattern': r'(FileNotFoundError|not found|cannot find|No such file|missing file)',
            'title': '파일 없음',
            'description': '필요한 파일을 찾을 수 없습니다. 게임 파일이 손상되었거나 불완전하게 설치되었습니다.',
            'severity': 'HIGH',
            'solutions': [
                '1. 마인크래프트를 재설치해보세요',
                '2. 게임 폴더의 파일이 손상되지 않았는지 확인하세요',
                '3. 안티바이러스 프로그램이 파일을 삭제하지 않았는지 확인하세요',
                '4. 런처의 "게임 파일 복구" 기능을 사용해보세요',
                '5. 모드 파일이 제대로 설치되었는지 확인하세요'
            ]
        },
        'permission_error': {
            'pattern': r'(Permission denied|Access denied|cannot write|cannot read|권한)',
            'title': '권한 오류',
            'description': '파일 접근 권한이 없습니다. 게임이 필요한 파일을 읽거나 쓸 수 없습니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 마인크래프트를 관리자 권한으로 실행해보세요',
                '   - 바탕화면 아이콘 우클릭 > 관리자 권한으로 실행',
                '2. 게임 폴더의 권한을 확인하세요',
                '3. 안티바이러스 프로그램의 차단을 확인하세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 게임 폴더를 다른 위치로 이동해보세요'
            ]
        },
        'texture_error': {
            'pattern': r'(missing texture|texture not found|resource pack error|texture pack)',
            'title': '텍스처 오류',
            'description': '텍스처 팩이나 리소스 팩에 문제가 있습니다. 게임 그래픽이 제대로 표시되지 않습니다.',
            'severity': 'LOW',
            'solutions': [
                '1. 텍스처 팩을 비활성화해보세요',
                '2. 텍스처 팩을 다시 설치해보세요',
                '3. 텍스처 팩이 마인크래프트 버전과 맞는지 확인하세요',
                '4. 손상된 텍스처 팩을 삭제하세요',
                '5. 기본 텍스처로 게임을 시작해보세요'
            ]
        },
        'launcher_error': {
            'pattern': r'(launcher error|login failed|authentication failed|account error)',
            'title': '런처 오류',
            'description': '마인크래프트 런처에 문제가 있습니다. 로그인이 안 되거나 런처가 제대로 작동하지 않습니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 마인크래프트 런처를 재시작하세요',
                '2. 마인크래프트 계정으로 다시 로그인하세요',
                '3. 런처를 완전히 제거하고 재설치하세요',
                '4. 인터넷 연결을 확인하세요',
                '5. 마인크래프트 계정 비밀번호를 확인하세요'
            ]
        },
        'mod_loader_error': {
            'pattern': r'(Forge|Fabric|loader error|bootstrap error|tweak class)',
            'title': '모드 로더 오류',
            'description': '모드 로더 (Forge/Fabric) 설치에 문제가 있습니다. 모드를 로드할 수 없습니다.',
            'severity': 'HIGH',
            'solutions': [
                '1. 모드 로더를 다시 설치해보세요',
                '2. 모드 로더 버전이 마인크래프트 버전과 맞는지 확인하세요',
                '3. 모드 로더 설치 파일을 다시 다운로드하세요',
                '4. 게임 폴더를 백업하고 처음부터 설치해보세요',
                '5. 모드 로더 공식 사이트에서 최신 버전을 다운로드하세요'
            ]
        },
        'chunk_loading_error': {
            'pattern': r'(chunk loading|chunk error|terrain generation|chunk data)',
            'title': '청크 로딩 오류',
            'description': '게임 월드의 청크를 로드하는 중 오류가 발생했습니다. 특정 위치에서 게임이 튕기거나 렉이 심합니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 렌더 거리를 줄여보세요 (설정 > 비디오 설정 > 렌더 거리: 8-10)',
                '2. 메모리를 늘려보세요',
                '3. 불필요한 모드를 제거해보세요',
                '4. 월드를 새로 만들어보세요',
                '5. 문제가 있는 청크를 삭제해보세요'
            ]
        },
        'crash_loop': {
            'pattern': r'(crash loop|repeatedly crash|infinite crash|keeps crashing)',
            'title': '반복 크래시',
            'description': '게임이 계속 튕기고 있습니다. 게임을 시작하자마자 튕기거나 특정 작업을 할 때마다 튕깁니다.',
            'severity': 'CRITICAL',
            'solutions': [
                '1. 세이프 모드로 게임을 시작해보세요',
                '2. 최근에 설치한 모드를 모두 제거해보세요',
                '3. 마인크래프트를 완전히 재설치해보세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 안티바이러스 프로그램을 비활성화해보세요'
            ]
        },
        'lag_freeze': {
            'pattern': r'(lag spike|freeze|stutter|slow performance|TPS)',
            'title': '렉/프리징',
            'description': '게임이 느리거나 멈춥니다. 프레임이 떨어지거나 게임이 반응하지 않습니다.',
            'severity': 'MEDIUM',
            'solutions': [
                '1. 렌더 거리를 줄여보세요 (설정 > 비디오 설정 > 렌더 거리: 8-10)',
                '2. 그래픽 설정을 낮춰보세요 (설정 > 비디오 설정 > 그래픽: 빠름)',
                '3. 메모리를 늘려보세요',
                '4. 불필요한 프로그램을 종료하세요',
                '5. 모드를 줄여보세요',
                '6. 파티클 효과를 줄여보세요 (설정 > 비디오 설정 > 파티클: 최소)'
            ]
        },
        'sound_error': {
            'pattern': r'(sound error|audio error|speaker error|sound device)',
            'title': '사운드 오류',
            'description': '게임 사운드에 문제가 있습니다. 소리가 안 나거나 이상한 소리가 납니다.',
            'severity': 'LOW',
            'solutions': [
                '1. 게임 설정에서 사운드 볼륨을 확인하세요',
                '2. 컴퓨터 사운드 설정을 확인하세요',
                '3. 오디오 드라이버를 업데이트하세요',
                '4. 마인크래프트를 재시작해보세요',
                '5. 게임 설정에서 사운드를 비활성화했는지 확인하세요'
            ]
        },
        'controller_error': {
            'pattern': r'(controller error|gamepad error|joystick error|input error)',
            'title': '컨트롤러 오류',
            'description': '게임 컨트롤러 인식에 문제가 있습니다. 컨트롤러가 작동하지 않습니다.',
            'severity': 'LOW',
            'solutions': [
                '1. 컨트롤러를 다시 연결해보세요',
                '2. 컨트롤러 드라이버를 업데이트하세요',
                '3. 게임 설정에서 컨트롤러를 다시 설정해보세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 다른 USB 포트에 연결해보세요'
            ]
        },
        'skin_error': {
            'pattern': r'(skin error|skin download|profile error|texture download)',
            'title': '스킨 오류',
            'description': '플레이어 스킨을 로드할 수 없습니다. 스킨이 표시되지 않거나 다운로드 오류가 발생합니다.',
            'severity': 'LOW',
            'solutions': [
                '1. 인터넷 연결을 확인하세요',
                '2. 마인크래프트 계정을 다시 로그인하세요',
                '3. 마인크래프트 런처를 재시작하세요',
                '4. 스킨을 다시 설정해보세요',
                '5. 기본 스킨으로 변경해보세요'
            ]
        },
        'mod_conflict': {
            'pattern': r'(Mixin|mixin|conflict|duplicate|already registered|incompatible mod)',
            'title': '모드 충돌',
            'description': '설치된 모드들이 서로 충돌하고 있습니다.',
            'severity': 'HIGH',
            'solutions': [
                '1. 최근에 설치한 모드를 제거해보세요',
                '2. 모드 버전이 마인크래프트 버전과 맞는지 확인하세요',
                '3. 모드 의존성을 확인하세요 (필요한 라이브러리 모드)',
                '4. 모드 폴더를 백업하고 처음부터 설치해보세요',
                '5. 모드 로더 (Forge, Fabric) 버전을 확인하세요',
                '6. 모드 개발자 페이지에서 호환성 정보를 확인하세요'
            ]
        },
        'graphics_error': {
            'pattern': r'(OpenGL|graphics|render|shader|display|video|GLFW|GLX)',
            'title': '그래픽 오류',
            'description': '그래픽 카드나 드라이버 문제입니다.',
            'solutions': [
                '1. 그래픽 드라이버를 최신 버전으로 업데이트하세요',
                '   - NVIDIA: nvidia.com에서 드라이버 다운로드',
                '   - AMD: amd.com에서 드라이버 다운로드',
                '   - Intel: intel.com에서 드라이버 다운로드',
                '2. 게임 설정에서 그래픽 옵션을 낮춰보세요',
                '3. 셰이더를 비활성화해보세요',
                '4. 마인크래프트를 재설치해보세요',
                '5. 그래픽 카드 드라이버를 완전히 제거하고 재설치해보세요'
            ]
        },
        'java_error': {
            'pattern': r'(java\.lang\.|Exception|Error|NullPointerException)',
            'title': '자바 오류',
            'description': '자바 실행 중 오류가 발생했습니다.',
            'solutions': [
                '1. 자바를 최신 버전으로 업데이트하세요',
                '   - java.com에서 최신 자바 다운로드',
                '2. 마인크래프트 런처를 재시작하세요',
                '3. 마인크래프트를 재설치해보세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 자바를 완전히 제거하고 재설치해보세요',
                '6. 런처 설정에서 자바 경로를 확인하세요'
            ]
        },
        'world_corruption': {
            'pattern': r'(corrupt|invalid|NBT|region|chunk|damaged)',
            'title': '월드 손상',
            'description': '저장된 월드 파일이 손상되었습니다.',
            'solutions': [
                '1. 백업 폴더에서 이전 버전의 월드를 복구해보세요',
                '   - .minecraft/saves 폴더에서 월드 폴더 찾기',
                '2. 문제가 있는 청크를 삭제해보세요',
                '   - MCEdit 또는 WorldEdit 모드 사용',
                '3. 월드를 새로 만들어보세요',
                '4. 월드 백업 파일을 확인하세요',
                '5. 월드 파일을 다른 컴퓨터에서 열어보세요'
            ]
        },
        'network_error': {
            'pattern': r'(Connection|socket|network|timeout|refused|unreachable)',
            'title': '네트워크 오류',
            'description': '서버 연결에 문제가 있습니다.',
            'solutions': [
                '1. 인터넷 연결을 확인하세요',
                '2. 방화벽 설정을 확인하세요',
                '   - Windows Defender 방화벽에서 마인크래프트 허용',
                '3. 서버가 온라인 상태인지 확인하세요',
                '4. 라우터를 재부팅해보세요',
                '5. 서버 주소와 포트 번호를 다시 확인하세요',
                '6. VPN을 사용 중이면 비활성화해보세요'
            ]
        },
        'version_mismatch': {
            'pattern': r'(version|mismatch|incompatible|protocol|outdated)',
            'title': '버전 불일치',
            'description': '클라이언트와 서버의 버전이 맞지 않습니다.',
            'solutions': [
                '1. 마인크래프트 버전을 확인하세요',
                '2. 서버 버전과 클라이언트 버전을 맞추세요',
                '3. 모드 버전을 확인하세요',
                '4. 런처에서 올바른 버전을 선택했는지 확인하세요',
                '5. 서버 관리자에게 서버 버전을 물어보세요',
                '6. 모드 로더 버전도 맞는지 확인하세요'
            ]
        },
        'file_not_found': {
            'pattern': r'(FileNotFoundError|not found|cannot find|No such file)',
            'title': '파일 없음',
            'description': '필요한 파일을 찾을 수 없습니다.',
            'solutions': [
                '1. 마인크래프트를 재설치해보세요',
                '2. 게임 폴더의 파일이 손상되지 않았는지 확인하세요',
                '3. 안티바이러스 프로그램이 파일을 삭제하지 않았는지 확인하세요',
                '4. 런처의 "게임 파일 복구" 기능을 사용해보세요',
                '5. 모드 파일이 제대로 설치되었는지 확인하세요'
            ]
        },
        'permission_error': {
            'pattern': r'(Permission|Access denied|권한|cannot write)',
            'title': '권한 오류',
            'description': '파일 접근 권한이 없습니다.',
            'solutions': [
                '1. 마인크래프트를 관리자 권한으로 실행해보세요',
                '   - 바탕화면 아이콘 우클릭 > 관리자 권한으로 실행',
                '2. 게임 폴더의 권한을 확인하세요',
                '3. 안티바이러스 프로그램의 차단을 확인하세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 게임 폴더를 다른 위치로 이동해보세요'
            ]
        },
        'texture_error': {
            'pattern': r'(texture|missing texture|texture pack|resource pack)',
            'title': '텍스처 오류',
            'description': '텍스처 팩이나 리소스 팩에 문제가 있습니다.',
            'solutions': [
                '1. 텍스처 팩을 비활성화해보세요',
                '2. 텍스처 팩을 다시 설치해보세요',
                '3. 텍스처 팩이 마인크래프트 버전과 맞는지 확인하세요',
                '4. 손상된 텍스처 팩을 삭제하세요',
                '5. 기본 텍스처로 게임을 시작해보세요'
            ]
        },
        'launcher_error': {
            'pattern': r'(launcher|login|authentication|account)',
            'title': '런처 오류',
            'description': '마인크래프트 런처에 문제가 있습니다.',
            'solutions': [
                '1. 마인크래프트 런처를 재시작하세요',
                '2. 마인크래프트 계정으로 다시 로그인하세요',
                '3. 런처를 완전히 제거하고 재설치하세요',
                '4. 인터넷 연결을 확인하세요',
                '5. 마인크래프트 계정 비밀번호를 확인하세요'
            ]
        },
        'mod_loader_error': {
            'pattern': r'(Forge|Fabric|loader|bootstrap|tweak)',
            'title': '모드 로더 오류',
            'description': '모드 로더 (Forge/Fabric) 설치에 문제가 있습니다.',
            'solutions': [
                '1. 모드 로더를 다시 설치해보세요',
                '2. 모드 로더 버전이 마인크래프트 버전과 맞는지 확인하세요',
                '3. 모드 로더 설치 파일을 다시 다운로드하세요',
                '4. 게임 폴더를 백업하고 처음부터 설치해보세요',
                '5. 모드 로더 공식 사이트에서 최신 버전을 다운로드하세요'
            ]
        },
        'chunk_loading_error': {
            'pattern': r'(chunk|loading|unloading|terrain)',
            'title': '청크 로딩 오류',
            'description': '게임 월드의 청크를 로드하는 중 오류가 발생했습니다.',
            'solutions': [
                '1. 렌더 거리를 줄여보세요 (설정 > 비디오 설정)',
                '2. 메모리를 늘려보세요',
                '3. 불필요한 모드를 제거해보세요',
                '4. 월드를 새로 만들어보세요',
                '5. 문제가 있는 청크를 삭제해보세요'
            ]
        },
        'crash_loop': {
            'pattern': r'(crash|loop|repeatedly|infinite)',
            'title': '반복 크래시',
            'description': '게임이 계속 튕기고 있습니다.',
            'solutions': [
                '1. 세이프 모드로 게임을 시작해보세요',
                '2. 최근에 설치한 모드를 모두 제거해보세요',
                '3. 마인크래프트를 완전히 재설치해보세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 안티바이러스 프로그램을 비활성화해보세요'
            ]
        },
        'lag_freeze': {
            'pattern': r'(lag|freeze|stutter|slow|performance)',
            'title': '렉/프리징',
            'description': '게임이 느리거나 멈춥니다.',
            'solutions': [
                '1. 렌더 거리를 줄여보세요',
                '2. 그래픽 설정을 낮춰보세요',
                '3. 메모리를 늘려보세요',
                '4. 불필요한 프로그램을 종료하세요',
                '5. 모드를 줄여보세요',
                '6. 파티클 효과를 줄여보세요'
            ]
        },
        'sound_error': {
            'pattern': r'(sound|audio|speaker|microphone)',
            'title': '사운드 오류',
            'description': '게임 사운드에 문제가 있습니다.',
            'solutions': [
                '1. 게임 설정에서 사운드 볼륨을 확인하세요',
                '2. 컴퓨터 사운드 설정을 확인하세요',
                '3. 오디오 드라이버를 업데이트하세요',
                '4. 마인크래프트를 재시작해보세요',
                '5. 게임 설정에서 사운드를 비활성화했는지 확인하세요'
            ]
        },
        'controller_error': {
            'pattern': r'(controller|joystick|gamepad|input)',
            'title': '컨트롤러 오류',
            'description': '게임 컨트롤러 인식에 문제가 있습니다.',
            'solutions': [
                '1. 컨트롤러를 다시 연결해보세요',
                '2. 컨트롤러 드라이버를 업데이트하세요',
                '3. 게임 설정에서 컨트롤러를 다시 설정해보세요',
                '4. 컴퓨터를 재부팅해보세요',
                '5. 다른 USB 포트에 연결해보세요'
            ]
        },
        'skin_error': {
            'pattern': r'(skin|cape|profile|texture download)',
            'title': '스킨 오류',
            'description': '플레이어 스킨을 로드할 수 없습니다.',
            'solutions': [
                '1. 인터넷 연결을 확인하세요',
                '2. 마인크래프트 계정을 다시 로그인하세요',
                '3. 마인크래프트 런처를 재시작하세요',
                '4. 스킨을 다시 설정해보세요',
                '5. 기본 스킨으로 변경해보세요'
            ]
        }
    }

    def __init__(self):
        self.errors = []
        self.file_path = ""
        self.lines = []
        self.ai_classifier = AIErrorClassifier()

    def analyze_file(self, file_path: str) -> dict:
        self.file_path = file_path
        self.errors = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.lines = f.readlines()
        except Exception as e:
            return {"error": f"파일을 읽을 수 없습니다: {e}"}

        if not self.lines:
            return {"error": "파일이 비어있습니다"}

        self._analyze_minecraft_errors()

        return {
            "file": file_path,
            "errors": self.errors,
            "total_lines": len(self.lines),
            "error_count": len(self.errors)
        }

    def _analyze_minecraft_errors(self):
        content = '\n'.join(self.lines)
        found_errors = set()
        error_scores = {}

        # 정규식 기반 분석
        for error_key, error_info in self.MINECRAFT_ERRORS.items():
            if error_key in found_errors:
                continue

            matches = []
            for i, line in enumerate(self.lines, 1):
                if re.search(error_info['pattern'], line, re.IGNORECASE):
                    match_obj = re.search(error_info['pattern'], line, re.IGNORECASE)
                    match_text = match_obj.group(0)
                    accuracy = len(match_text) / len(line) if line else 0
                    matches.append((i, line.strip(), accuracy, match_text))

            if matches:
                best_match = max(matches, key=lambda x: (x[2], len(x[1])))
                line_num, line_content, accuracy, match_text = best_match

                importance = self._calculate_importance(error_key, accuracy, len(matches))

                self.errors.append({
                    'type': error_key,
                    'title': error_info['title'],
                    'description': error_info['description'],
                    'solutions': error_info['solutions'],
                    'severity': error_info.get('severity', 'MEDIUM'),
                    'line': line_num,
                    'content': line_content[:150],
                    'match_count': len(matches),
                    'accuracy': round(accuracy * 100, 1),
                    'importance': importance,
                    'matched_text': match_text,
                    'detection_method': 'Regex'
                })
                error_scores[error_key] = importance
                found_errors.add(error_key)

        # AI 기반 분석 
        for i, line in enumerate(self.lines, 1):
            if not line.strip():
                continue
            
            error_category, confidence = self.ai_classifier.predict_error_type(line)
            
            if error_category and error_category not in found_errors:
                error_info = self.MINECRAFT_ERRORS.get(error_category)
                if error_info:
                    ai_importance = int((confidence / 100) * 70)  # AI는 최대 70점
                    
                    self.errors.append({
                        'type': error_category,
                        'title': error_info['title'],
                        'description': error_info['description'],
                        'solutions': error_info['solutions'],
                        'severity': error_info.get('severity', 'MEDIUM'),
                        'line': i,
                        'content': line.strip()[:150],
                        'match_count': 1,
                        'accuracy': confidence,
                        'importance': ai_importance,
                        'matched_text': line.strip()[:50],
                        'detection_method': f'AI ({confidence}%)'
                    })
                    found_errors.add(error_category)

        self.errors.sort(key=lambda x: x['importance'], reverse=True)

    def _calculate_importance(self, error_key: str, accuracy: float, match_count: int) -> int:
        """중요도 계산 (0-100)"""
        severity_weight = {
            'CRITICAL': 100,
            'HIGH': 80,
            'MEDIUM': 50,
            'LOW': 20
        }

        severity = self.MINECRAFT_ERRORS[error_key].get('severity', 'MEDIUM')
        base_score = severity_weight.get(severity, 50)

        accuracy_score = accuracy * 20
        match_score = min(match_count * 2, 10)

        total = base_score + accuracy_score + match_score
        return min(int(total), 100)


class MinecraftCrashGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("마인크래프트 크래시 분석기")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1e1e1e")
        
        self.analyzer = MinecraftCrashAnalyzer()
        self.current_results = None
        
        self._create_widgets()
        self._apply_styles()

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 다크 테마
        style.configure('TFrame', background="#1e1e1e")
        style.configure('TLabel', background="#1e1e1e", foreground="#ffffff")
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background="#1e1e1e", foreground="#00ff00")
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'), background="#2d2d2d", foreground="#ffff00")
        style.configure('TButton', font=('Arial', 10))
        style.configure('TLabelframe', background="#1e1e1e", foreground="#ffffff")
        style.configure('TLabelframe.Label', background="#1e1e1e", foreground="#ffffff")

    def _create_widgets(self):
        # 메뉴바
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="로그 파일 열기", command=self._select_file)
        file_menu.add_command(label="결과 저장", command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)
        
        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="사용 방법", command=self._show_help)
        help_menu.add_command(label="정보", command=self._show_about)
        
        # 상단 프레임
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=15, pady=15)

        ttk.Label(top_frame, text="⛏️ 마인크래프트 크래시 분석기", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(top_frame, text="게임이 튕겼을 때 로그를 분석해서 문제를 찾아줍니다", 
                 background="#1e1e1e", foreground="#888888").pack(side=tk.LEFT, padx=20)

        # 버튼 프레임
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=15, pady=10)

        ttk.Button(button_frame, text="📁 로그 파일 선택", command=self._select_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 새로고침", command=self._refresh).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 결과 저장", command=self._save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🗑️ 초기화", command=self._clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❓ 도움말", command=self._show_help).pack(side=tk.LEFT, padx=5)

        # 정보 프레임
        info_frame = ttk.LabelFrame(self.root, text="📋 파일 정보", padding=10)
        info_frame.pack(fill=tk.X, padx=15, pady=5)

        self.info_text = tk.StringVar(value="로그 파일을 선택하세요")
        ttk.Label(info_frame, textvariable=self.info_text, foreground="#00ff00").pack(anchor=tk.W)

        # 메인 콘텐츠 프레임
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # 왼쪽: 오류 목록
        left_frame = ttk.LabelFrame(content_frame, text="🔴 발견된 문제", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        self.error_listbox = tk.Listbox(left_frame, height=25, width=30, 
                                        bg="#2d2d2d", fg="#ffffff", font=('Arial', 9))
        self.error_listbox.pack(fill=tk.BOTH, expand=True)
        self.error_listbox.bind('<<ListboxSelect>>', self._on_error_select)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.error_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.error_listbox.config(yscrollcommand=scrollbar.set)

        # 오른쪽: 상세 정보
        right_frame = ttk.LabelFrame(content_frame, text="📖 상세 설명 및 해결 방법", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.detail_text = scrolledtext.ScrolledText(right_frame, height=25, width=60, 
                                                     font=('Courier', 10), bg="#2d2d2d", fg="#ffffff")
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        self.detail_text.config(state=tk.DISABLED)

    def _select_file(self):
        file_path = filedialog.askopenfilename(
            title="마인크래프트 로그 파일 선택",
            filetypes=[
                ("로그 및 텍스트 파일", "*.log *.txt"),
                ("로그 파일", "*.log"),
                ("텍스트 파일", "*.txt"),
                ("크래시 리포트", "crash-*.txt"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            self._analyze_file(file_path)

    def _analyze_file(self, file_path):
        def analyze():
            self.info_text.set("분석 중...")
            self.root.update()
            
            results = self.analyzer.analyze_file(file_path)
            self.current_results = results
            
            self._display_results(results)
            self.info_text.set(f"✅ 분석 완료: {Path(file_path).name}")

        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()

    def _display_results(self, results):
        self.error_listbox.delete(0, tk.END)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)

        if results.get('error'):
            self.error_listbox.insert(tk.END, "❌ 오류 발생")
            self.detail_text.insert(tk.END, f"오류: {results['error']}\n")
        elif results['error_count'] == 0:
            self.error_listbox.insert(tk.END, "✅ 문제 없음")
            self.detail_text.insert(tk.END, "로그에서 알려진 문제를 찾지 못했습니다.\n\n")
            self.detail_text.insert(tk.END, "💡 팁:\n")
            self.detail_text.insert(tk.END, "- 마인크래프트 런처를 재시작해보세요\n")
            self.detail_text.insert(tk.END, "- 컴퓨터를 재부팅해보세요\n")
            self.detail_text.insert(tk.END, "- 마인크래프트를 재설치해보세요\n")
        else:
            # 심각도별로 정렬
            severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            sorted_errors = sorted(
                results['errors'],
                key=lambda x: severity_order.get(x.get('severity', 'MEDIUM'), 2)
            )
            
            for error in sorted_errors:
                severity = error.get('severity', 'MEDIUM')
                importance = error.get('importance', 50)
                accuracy = error.get('accuracy', 0)
                
                severity_icon = {
                    'CRITICAL': '[🔴🔴🔴]',
                    'HIGH': '[🔴🔴]',
                    'MEDIUM': '[🟡]',
                    'LOW': '[🟢]'
                }
                icon = severity_icon.get(severity, '[🟡]')
                
                # 중요도 바 표시
                bar_length = int(importance / 10)
                bar = '█' * bar_length + '░' * (10 - bar_length)
                
                self.error_listbox.insert(tk.END, f"{icon} {error['title']} [{bar}] {importance}%")

        self.detail_text.config(state=tk.DISABLED)

    def _on_error_select(self, event):
        selection = self.error_listbox.curselection()
        if not selection or not self.current_results:
            return

        index = selection[0]
        error = self.current_results['errors'][index]

        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)

        # 심각도 아이콘
        severity_icon = {
            'CRITICAL': '[🔴🔴🔴]',
            'HIGH': '[🔴🔴]',
            'MEDIUM': '[🟡]',
            'LOW': '[🟢]'
        }
        severity = error.get('severity', 'MEDIUM')
        icon = severity_icon.get(severity, '[🟡]')

        # 제목
        self.detail_text.insert(tk.END, f"{icon} {error['title']}\n")
        self.detail_text.insert(tk.END, "=" * 70 + "\n\n")

        # 심각도 표시
        severity_text = {
            'CRITICAL': '⚠️ 매우 심각 - 즉시 해결 필요',
            'HIGH': '⚠️ 심각 - 빠른 해결 필요',
            'MEDIUM': '⚠️ 중간 - 해결 권장',
            'LOW': 'ℹ️ 낮음 - 선택적 해결'
        }
        self.detail_text.insert(tk.END, f"심각도: {severity_text.get(severity, '알 수 없음')}\n")
        
        # 중요도 표시
        importance = error.get('importance', 50)
        accuracy = error.get('accuracy', 0)
        detection_method = error.get('detection_method', 'Unknown')
        bar_length = int(importance / 10)
        bar = '█' * bar_length + '░' * (10 - bar_length)
        self.detail_text.insert(tk.END, f"중요도: [{bar}] {importance}% (정확도: {accuracy}%)\n")
        self.detail_text.insert(tk.END, f"감지 방법: {detection_method}\n\n")

        # 설명
        self.detail_text.insert(tk.END, "📝 문제 설명:\n")
        self.detail_text.insert(tk.END, f"{error['description']}\n\n")

        # 발견 위치
        self.detail_text.insert(tk.END, "📍 발견 위치:\n")
        self.detail_text.insert(tk.END, f"줄 {error['line']}: {error['content']}\n\n")

        # 해결 방법
        self.detail_text.insert(tk.END, "✅ 해결 방법 (순서대로 시도하세요):\n")
        for solution in error['solutions']:
            self.detail_text.insert(tk.END, f"{solution}\n")

        self.detail_text.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.detail_text.insert(tk.END, f"💡 팁: 위의 해결 방법을 순서대로 시도해보세요.\n")
        self.detail_text.insert(tk.END, f"문제가 해결되지 않으면 다음 방법을 시도하세요.\n")

        self.detail_text.config(state=tk.DISABLED)

    def _refresh(self):
        if self.current_results:
            self._display_results(self.current_results)
            messagebox.showinfo("완료", "새로고침 완료")

    def _save_results(self):
        if not self.current_results:
            messagebox.showwarning("경고", "분석 결과가 없습니다")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    results = self.current_results
                    f.write("=" * 80 + "\n")
                    f.write("⛏️ 마인크래프트 크래시 분석 결과\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write(f"파일: {results['file']}\n")
                    f.write(f"총 줄: {results['total_lines']}\n")
                    f.write(f"발견된 문제: {results['error_count']}개\n\n")
                    
                    if results['error_count'] == 0:
                        f.write("✅ 알려진 문제를 찾지 못했습니다.\n\n")
                        f.write("💡 팁:\n")
                        f.write("- 마인크래프트 런처를 재시작해보세요\n")
                        f.write("- 컴퓨터를 재부팅해보세요\n")
                        f.write("- 마인크래프트를 재설치해보세요\n")
                    else:
                        # 심각도별로 정렬
                        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
                        sorted_errors = sorted(
                            results['errors'],
                            key=lambda x: severity_order.get(x.get('severity', 'MEDIUM'), 2)
                        )
                        
                        for error in sorted_errors:
                            severity = error.get('severity', 'MEDIUM')
                            severity_text = {
                                'CRITICAL': '🔴🔴🔴 매우 심각',
                                'HIGH': '🔴🔴 심각',
                                'MEDIUM': '🟡 중간',
                                'LOW': '🟢 낮음'
                            }
                            
                            f.write(f"\n{'='*80}\n")
                            f.write(f"🔴 {error['title']}\n")
                            f.write(f"심각도: {severity_text.get(severity, '알 수 없음')}\n")
                            f.write(f"{'='*80}\n\n")
                            
                            f.write(f"📝 문제 설명:\n{error['description']}\n\n")
                            
                            f.write(f"📍 발견 위치:\n줄 {error['line']}: {error['content']}\n\n")
                            
                            f.write(f"✅ 해결 방법 (순서대로 시도하세요):\n")
                            for solution in error['solutions']:
                                f.write(f"{solution}\n")
                            f.write("\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("📌 주의사항:\n")
                    f.write("- 위의 해결 방법을 순서대로 시도해보세요\n")
                    f.write("- 문제가 해결되지 않으면 다음 방법을 시도하세요\n")
                    f.write("- 마인크래프트 공식 포럼에서 도움을 받을 수 있습니다\n")
                    f.write("=" * 80 + "\n")
                
                messagebox.showinfo("완료", f"결과가 저장되었습니다:\n{file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"저장 실패: {e}")

    def _clear(self):
        self.error_listbox.delete(0, tk.END)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.config(state=tk.DISABLED)
        self.current_results = None
        self.info_text.set("로그 파일을 선택하세요")

    def _show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("사용 방법")
        help_window.geometry("700x600")
        help_window.configure(bg="#1e1e1e")
        
        help_text = scrolledtext.ScrolledText(help_window, font=('Arial', 10), 
                                             bg="#2d2d2d", fg="#ffffff", wrap=tk.WORD)
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        help_text.config(state=tk.NORMAL)
        
        help_content = """📖 마인크래프트 크래시 분석기 - 사용 방법

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 기본 사용법
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 로그 파일 선택
   - "📁 로그 파일 선택" 버튼을 클릭하세요
   - 마인크래프트 크래시 로그 파일을 선택하세요
   - 지원 형식: .log, .txt, crash-*.txt

2. 분석 결과 확인
   - 왼쪽에 발견된 문제 목록이 표시됩니다
   - 각 문제는 심각도와 중요도로 표시됩니다
   - 문제를 클릭하면 오른쪽에 상세 정보가 표시됩니다

3. 해결 방법 확인
   - 상세 정보에서 해결 방법을 확인하세요
   - 위에서부터 순서대로 시도해보세요
   - 한 가지씩 시도하고 테스트하세요

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 심각도 표시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[🔴🔴🔴] CRITICAL (매우 심각)
  - 즉시 해결이 필요합니다
  - 게임을 할 수 없는 상태입니다

[🔴🔴] HIGH (심각)
  - 빠른 해결이 필요합니다
  - 게임 플레이에 큰 문제가 있습니다

[🟡] MEDIUM (중간)
  - 해결을 권장합니다
  - 게임 플레이에 약간의 문제가 있습니다

[🟢] LOW (낮음)
  - 선택적으로 해결할 수 있습니다
  - 게임 플레이에 미미한 영향을 줍니다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 중요도 표시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

중요도는 0-100% 범위로 표시됩니다:

[██████████] 100% - 매우 중요 (즉시 해결)
[████████░░] 80% - 중요 (빠른 해결)
[██████░░░░] 60% - 보통 (해결 권장)
[████░░░░░░] 40% - 낮음 (선택적 해결)

중요도는 다음 요소로 계산됩니다:
- 심각도 (기본 점수)
- 정확도 (패턴 매칭 정확도)
- 매치 개수 (같은 오류가 여러 번 나타나는 경우)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 결과 저장
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- "💾 결과 저장" 버튼을 클릭하세요
- 분석 결과를 텍스트 파일로 저장할 수 있습니다
- 나중에 참고하거나 다른 사람과 공유할 수 있습니다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 새로고침 및 초기화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- "🔄 새로고침": 현재 분석 결과를 다시 표시합니다
- "🗑️ 초기화": 모든 결과를 지우고 초기 상태로 돌아갑니다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 팁
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 중요도가 높은 오류부터 해결하세요
2. 정확도를 확인하고 참고하세요
3. 해결 방법을 순서대로 시도하세요
4. 한 가지씩 시도하고 테스트하세요
5. 문제가 해결되면 멈추세요

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ 문제가 해결되지 않으면?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 마인크래프트 공식 포럼에서 도움을 받으세요
2. 마인크래프트 커뮤니티에 질문하세요
3. 모드 개발자에게 문의하세요
4. 컴퓨터를 재부팅해보세요
5. 마인크래프트를 완전히 재설치해보세요

행운을 빕니다! 즐거운 마인크래프트 플레이 되세요! ⛏️
"""
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)

    def _show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("정보")
        about_window.geometry("500x400")
        about_window.configure(bg="#1e1e1e")
        
        about_text = scrolledtext.ScrolledText(about_window, font=('Arial', 11), 
                                              bg="#2d2d2d", fg="#ffffff", wrap=tk.WORD)
        about_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        about_text.config(state=tk.NORMAL)
        
        about_content = """⛏️ 마인크래프트 크래시 분석기

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 프로그램 정보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이 프로그램은 마인크래프트 게임이 튕겼을 때 
로그 파일을 분석해서 문제를 찾아주고 
해결 방법을 제시해주는 도구입니다.

컴맹도 쉽게 사용할 수 있도록 
한국어로 작성되었습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ 주요 기능
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 18가지 오류 패턴 감지
✓ 정확도 기반 오류 분석 (0-100%)
✓ 중요도 시스템 (0-100점)
✓ 심각도 4단계 분류
✓ 상세한 설명 및 해결 방법
✓ 발견 위치 표시
✓ 결과 저장 기능

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👨‍💻 제작자
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

제작자: Dangel

이 프로그램은 마인크래프트 플레이어들을 위해 
만들어졌습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🙏 감사합니다!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이 프로그램을 사용해주셔서 감사합니다.

행운을 빕니다! 즐거운 마인크래프트 플레이 되세요! ⛏️
"""
        
        about_text.insert(tk.END, about_content)
        about_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = MinecraftCrashGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
