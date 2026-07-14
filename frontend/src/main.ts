import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app'; // 確保指向 app.ts 裡的 AppComponent

bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));
