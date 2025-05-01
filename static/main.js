import { createApp } from 'vue'
import { createPinia } from '@pinia/nuxt'
import App from './App.vue'
import vuetify from './plugins/vuetify'

const pinia = createPinia()
const app = createApp(App)

app.use(vuetify)
app.use(pinia)

app.mount('#app')
