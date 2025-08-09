#!/usr/bin/env python
"""
sessions_plotter.py has 3 functions to create spectogram, histogram, 2d_histogram from [(ts, size),..] session.
"""

import matplotlib.pyplot as plt
import numpy as np

MTU = 1500

"""
ts: 각 패킷 도착시간(초단위, 오름차순)
sizes: 각 패킷의 길ㅇ(바이트)
name: 그래프 제목에 쓸 문자열(옵션)
"""
def session_spectogram(ts, sizes, name=None):
    # 점(.)으로 산점도 그리기
    # 시간에 따라 어떤 크기의 패킷이 언제 도착했는지 시각적으로 확인 가능
    plt.scatter(ts, sizes, marker='.')
    # y축의 범위를 0에서 MTU(1500 bytes)로 고정 (이더넷 MTU 기준)
    plt.ylim(0, MTU)
    # x축은 세션의 시작~끝 시간
    plt.xlim(ts[0], ts[-1])

    # plt.yticks(np.arange(0, MTU, 10))
    # plt.xticks(np.arange(int(ts[0]), int(ts[-1]), 10))

    # 매개변수로 이름 주어지면 세션 정보가 제목에 포함됨
    plt.title(name + " Session Spectogram")
    # x축 단위는 초, y축 단위는 bytes
    plt.ylabel('Size [B]')
    plt.xlabel('Time [sec]')

    # 격자 표시
    plt.grid(True)
    # 화면에 출력함. 리턴값 없음
    plt.show()


"""
fpath: 결과 이미지를 저장할 파일 경로
show: True이면 화면에 표시, False면 화면에 표시 안함
tps: 세션의 총 시간(초). 주어지면 시간 정규화 시 이 값 사용, 없으면 ts[-1]-ts[0] 사용
"""
def session_article_spectogram(ts, sizes, fpath=None, show=True, tps=None):
    # 정규화 기준 설정
    if tps is None:
        max_delta_time = ts[-1] - ts[0]
    else:
        max_delta_time = tps

    """
    시간 정규화:
    - 시간 값을 0 ~ MTU(1500) 범위로 선형 변환
    - 그 결과, x축과 y축 둘 다 0~1500 픽셀 범위로 동일해져서, 이후 CNN 입력 이미지와 좌표 체계가 일치 
    """
    ts_norm = ((np.array(ts) - ts[0]) / max_delta_time) * MTU

    # figure()로 새로운 그림 객체를 선언함. 빈 캔버스 생성한다.
    plt.figure()
    # 마커 크기는 s=5로, 색상은 검정 c='k'로.
    plt.scatter(ts_norm, sizes, marker=',', c='k', s=5)
    plt.ylim(0, MTU)
    plt.xlim(0, MTU)
    plt.ylabel('Packet Size [B]')
    plt.xlabel('Normalized Arrival Time')
    # 색상 맵은 binary(흑백)으로. 논문 Figure 스타일과 유사함
    plt.set_cmap('binary')
    plt.axes().set_aspect('equal')
    # 깔끔한 자료 스타일인 경우는 grid를 false로 설정함
    plt.grid(False)
    if fpath is not None:
        # plt.savefig(OUTPUT_DIR + fname, bbox_inches='tight', pad_inches=1)
        plt.savefig(fpath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

"""
패킷 크기 분포를 계산하는 함수
- size: 각 패킷 길이 배열(int, 바이트 단위)
- plot: True이면 히스토그램을 시각화, False면 값만 반환
"""
def session_histogram(sizes, plot=False):
    """
    np.histogram
    : sizes 배열을 bins 구간별로 나눠 각 구간에 들어있는 값의 개수를 셈
    - bins=range(0, MTU + 1, 1): 0 ~ 1500 byte를 1바이트 단위로 나눔
    - 결과:
        - hist: 각 구간에 해당하는 패킷 개수
        - bin_edges: 각 구간의 경계값 배열
    """
    hist, bin_edges = np.histogram(sizes, bins=range(0, MTU + 1, 1))

    # 시각화 옵션
    if plot:
        """
        bin_edges[:-1]: 각 구간의 시작값
        width=1: 막대 폭 1 (바이트 단위)
        xlim: x축 범위를 데이터 크기 범위에 맞춰 설정
        """
        plt.bar(bin_edges[:-1], hist, width=1)
        plt.xlim(min(bin_edges), max(bin_edges)+100)
        plt.show()
    # 2 bytes 부호 없는 정수로 변환해서 반환
    return hist.astype(np.uint16)

"""
ts: 패킷 도착 시간 배열(초)
sizes: 패킷 크기 배열(바이트)
plot: True이면 2D 히트맵으로 시각화
tps: 세션의 총 길이(초). 없으면 실제 세션 길이(ts[-1] - ts[0])으로 사용

이 함수는 FlowPic 논문에서 말하는 CNN에 입력한 1500x1500 픽셀 이미지다.
"""
def session_2d_histogram(ts, sizes, plot=False, tps=None):
    """
    시간축을 0 ~ MTU(1500) 픽셀 범위로 맞추기 위한 기준값
    FlowPic 논문에서처럼 60초 -> 1500 픽셀로 매핑 가능 
    """
    if tps is None:
        max_delta_time = ts[-1] - ts[0]
    else:
        max_delta_time = tps

    # ts_norm = map(int, ((np.array(ts) - ts[0]) / max_delta_time) * MTU)
    ts_norm = ((np.array(ts) - ts[0]) / max_delta_time) * MTU
    H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=(range(0, MTU + 1, 1), range(0, MTU + 1, 1)))

    if plot:
        # 셀별 카운트를 색으로 표현
        plt.pcolormesh(xedges, yedges, H)
        plt.colorbar()
        plt.xlim(0, MTU)
        plt.ylim(0, MTU)
        plt.set_cmap('binary')
        plt.show()
    return H.astype(np.uint16)
