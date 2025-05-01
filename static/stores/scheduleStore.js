// Pinia store for browser (CDN/global Pinia API)
window.store = new Vuex.Store({
  state: {
    schedule: {},
    scheduleSettings: {
      startDate: '',
      duration: 21,
      generatedAt: null
    },
    loading: false,
    errorMessage: null
  },
  
  mutations: {
    setScheduleSettings(state, settings) {
      state.scheduleSettings = { ...state.scheduleSettings, ...settings };
      localStorage.setItem('scheduleSettings', JSON.stringify(state.scheduleSettings));
    },
    setSchedule(state, scheduleData) {
      console.log('[Vuex] setSchedule called:', scheduleData);
      state.schedule = scheduleData;
      localStorage.setItem('currentSchedule', JSON.stringify(scheduleData));
    },
    setErrorMessage(state, message) {
      state.errorMessage = message;
    },
    clearSchedule(state) {
      state.schedule = {};
      localStorage.removeItem('currentSchedule');
    }
  },
  actions: {
    updateScheduleSettings({ commit }, settings) {
      commit('setScheduleSettings', settings);
    },
    saveSchedule({ commit }, scheduleData) {
      try {
        console.log('[Vuex] saveSchedule action called:', scheduleData);
        commit('setSchedule', scheduleData);
      } catch (error) {
        console.error('Error saving schedule:', error);
        commit('setErrorMessage', 'Failed to save schedule');
        throw error;
      }
    },
    loadSchedule({ commit }) {
    // Load scheduleSettings from localStorage
    try {
      const savedSettings = localStorage.getItem('scheduleSettings');
      if (savedSettings) {
        console.log('[Vuex] loadSchedule: loaded scheduleSettings', savedSettings);
        commit('setScheduleSettings', JSON.parse(savedSettings));
      }
    } catch (error) {
      console.error('Error loading scheduleSettings:', error);
    }

    try {
      const savedSchedule = localStorage.getItem('currentSchedule');
      if (savedSchedule) {
        console.log('[Vuex] loadSchedule: loaded currentSchedule', savedSchedule);
        commit('setSchedule', JSON.parse(savedSchedule));
      } else {
        console.log('[Vuex] loadSchedule: no currentSchedule found in localStorage');
      }
    } catch (error) {
      console.error('Error loading schedule:', error);
      commit('setErrorMessage', 'Failed to load schedule');
      throw error;
    }
  },
    clearSchedule({ commit }) {
      commit('clearSchedule');
    }
  }
});
// Expose for browser global access
// window.store is now available for Vuex

