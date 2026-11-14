// Both pins bind the audited public projection to its immutable source catalog.
import {createHash} from 'node:crypto';
import {validatePlotCatalog} from './plot-gallery.js';
export function verifiedPlotProjection(raw){
  if(createHash('sha256').update(raw).digest('hex')!=='4f5395fcee736bc48734a9296952d5a2cef8a8b68a3fc8e732f92d0717b9b4a7')throw Error('Public plot projection changed; re-audit against the pinned CSV');
  return validatePlotCatalog(JSON.parse(raw));
}
