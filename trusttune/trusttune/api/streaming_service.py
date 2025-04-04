from fastapi import APIRouter, HTTPException
import numpy as np

router = APIRouter()

@router.post("/calibrators/{calibrator_id}/update")
async def update_streaming_calibrator(
    calibrator_id: str,
    request: dict
):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(status_code=404, detail="Calibrator not found")
    
    try:
        calibrator = streaming_calibrators[calibrator_id]
        scores = np.array(request["scores"], dtype=np.float64)
        labels = np.array(request["labels"], dtype=np.int32)
        
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have the same length")
            
        updated = calibrator.update(scores, labels)
        metrics = calibrator.get_metrics()
        
        return {
            "updated": updated,
            "version": calibrator.version,
            "metrics": metrics
        }
        
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 